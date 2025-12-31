//g++ -shared -fPIC -o libvmm_hook.so vmm_hook.cpp -I/usr/local/cuda/include -lcuda -ldl
#include <cuda.h>
#include <cuda_runtime.h>
#include <dlfcn.h>
#include <iostream>
#include <map>
#include <mutex>
#include <unistd.h>
#include <string>
#include <cstdlib>
#include <cstdio>

// 简单的全局记录
struct VmmBlock {
    CUmemGenericAllocationHandle handle;
    size_t size;
    int fd;
};

std::mutex g_mutex;
std::map<void*, VmmBlock> g_allocations;

typedef cudaError_t (*cudaFree_t)(void*);
cudaFree_t real_cudaFree = nullptr;

#define CHECK_CU(x) { \
    CUresult res = x; \
    if (res != CUDA_SUCCESS) { \
        const char* err; \
        cuGetErrorString(res, &err); \
        std::cerr << "[VMM_HOOK] Error: " << #x << " failed: " << err << std::endl; \
        return cudaErrorMemoryAllocation; \
    } \
}

extern "C" {

cudaError_t cudaMalloc(void** devPtr, size_t size) {
    // 1. 获取当前设备
    int current_device = 0;
    cudaGetDevice(&current_device);

    // 2. 准备 VMM 属性
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = current_device; 
    prop.win32HandleMetaData = NULL; 
    prop.requestedHandleTypes = CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR;

    size_t granularity;
    cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED);
    size_t padded_size = (size + granularity - 1) & ~(granularity - 1);

    // 3. VMM Reserve & Create & Map
    CUdeviceptr d_ptr;
    CHECK_CU(cuMemAddressReserve(&d_ptr, padded_size, 0, 0, 0));

    CUmemGenericAllocationHandle handle;
    CHECK_CU(cuMemCreate(&handle, padded_size, &prop, 0));

    CHECK_CU(cuMemMap(d_ptr, padded_size, 0, handle, 0));

    // ==========================================
    // [关键修改] 4. 设置访问权限 (Set Access)
    // 遍历所有 GPU，尝试开启访问权限。
    // 这样 Rank 0 也能写入 Rank 1 分配的 local_buff。
    // ==========================================
    int device_count = 0;
    cudaGetDeviceCount(&device_count);

    for (int i = 0; i < device_count; ++i) {
        CUmemAccessDesc desc = {};
        desc.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc.location.id = i; // 尝试给第 i 个设备授权
        desc.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        
        // 尝试设置权限
        CUresult res = cuMemSetAccess(d_ptr, padded_size, &desc, 1);
        
        // 如果是当前设备，必须成功；如果是其他设备(P2P)，失败了可以容忍(可能是硬件不支持)
        if (i == current_device) {
            CHECK_CU(res);
        }
    }

    // 5. 导出 FD
    int fd = -1;
    CHECK_CU(cuMemExportToShareableHandle(&fd, handle, CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR, 0));

    // 6. 记录 [同时写入 FD 和 Base Address]
    {
        std::lock_guard<std::mutex> lock(g_mutex);
        g_allocations[(void*)d_ptr] = {handle, padded_size, fd};
    }

    // 7. 输出 FD 到文件，供外部使用
    const char* rank_str = getenv("LOCAL_RANK");
    std::string filename;
    if (rank_str) filename = "/tmp/vmm_fd_rank_" + std::string(rank_str);
    else filename = "/tmp/vmm_fd_pid_" + std::to_string(getpid());
    
    FILE* fp = fopen(filename.c_str(), "w");
    if (fp) { 
        // 格式: "FD BaseAddr_Long"
        fprintf(fp, "%d %lu", fd, (uintptr_t)d_ptr); 
        fclose(fp); 
    }

    std::cout << "\n[VMM_HOOK] >>> Intercepted cudaMalloc(" << size << ")" << std::endl;
    // ...

    *devPtr = (void*)d_ptr;
    return cudaSuccess;
}


cudaError_t cudaFree(void* devPtr) {
    if (devPtr == nullptr) return cudaSuccess;

    std::lock_guard<std::mutex> lock(g_mutex);
    auto it = g_allocations.find(devPtr);

    if (it != g_allocations.end()) {
        VmmBlock block = it->second;
        cuMemUnmap((CUdeviceptr)devPtr, block.size);
        cuMemAddressFree((CUdeviceptr)devPtr, block.size);
        cuMemRelease(block.handle);
        close(block.fd);
        g_allocations.erase(it);
        return cudaSuccess;
    }

    if (!real_cudaFree) {
        real_cudaFree = (cudaFree_t)dlsym(RTLD_NEXT, "cudaFree");
    }
    if (real_cudaFree) {
        return real_cudaFree(devPtr);
    }
    return cudaSuccess;
}

}