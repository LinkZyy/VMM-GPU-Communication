#include <torch/extension.h>
#include <cuda.h>
// #include <cuda_runtime.h> // 彻底移除 Runtime 头文件依赖，保证纯粹性
#include <vector>
#include <iostream>
#include <stdexcept>

// ------------------------------------------------------------------
// Helper: 错误检查宏 (针对 Driver API)
// ------------------------------------------------------------------
#define CHECK_CU(x) { \
    CUresult res = x; \
    if (res != CUDA_SUCCESS) { \
        const char* err; \
        cuGetErrorString(res, &err); \
        std::cerr << "[VMM_IPC_ERROR] API Call Failed at line " << __LINE__ << std::endl; \
        std::cerr << "[VMM_IPC_ERROR] Code: " << res << " String: " << err << std::endl; \
        throw std::runtime_error(std::string("CUDA Driver Error: ") + err); \
    } \
}

// ------------------------------------------------------------------
// Helper: 空删除器
// ------------------------------------------------------------------
void nop_deleter(void* ptr) { 
    // Do nothing. 
    // Tensor 只是借用物理内存，生命周期由 VMM Handle 和接收端进程管理。
    // 千万不要 free，否则会把底层映射给拆了。
}

// ------------------------------------------------------------------
// Main Function: Import Tensor with Offset
// ------------------------------------------------------------------
torch::Tensor import_tensor_from_fd(int fd, 
                                    int64_t phy_size,       // 物理块的总大小 (e.g. 20GB)
                                    int64_t offset,         // 字节偏移量
                                    int mapping_device_id,  // 当前设备 (Rank 1)
                                    int resident_device_id, // 数据源设备 (Rank 0)
                                    std::vector<int64_t> shape, 
                                    c10::ScalarType dtype) {
    
    // 1. 获取 Handle
    CUmemGenericAllocationHandle handle;
    CHECK_CU(cuMemImportFromShareableHandle(&handle, 
                                            (void*)(uintptr_t)fd, 
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    // 2. Reserve VA (预留虚拟地址空间)
    // 注意：我们要映射的是整个物理块(phy_size)，以便覆盖后面可能的 offset
    CUdeviceptr d_ptr_base;
    CHECK_CU(cuMemAddressReserve(&d_ptr_base, phy_size, 0, 0, 0));

    // 3. Map (建立页表映射)
    // 将整个物理块映射到 Base Address
    CHECK_CU(cuMemMap(d_ptr_base, phy_size, 0, handle, 0));

    // 4. Set Access (设置权限 - 替代 cudaDeviceEnablePeerAccess)
    std::vector<CUmemAccessDesc> accessDescriptors;
    
    // 权限 A: Rank 1 (Receiver) 可读写
    CUmemAccessDesc desc1 = {};
    desc1.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc1.location.id = mapping_device_id;
    desc1.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDescriptors.push_back(desc1);

    // 权限 B: Rank 0 (Sender) 可读写
    // 这一步至关重要，允许 PyTorch 在源设备 stream 上操作这块内存
    if (mapping_device_id != resident_device_id) {
        CUmemAccessDesc desc0 = {};
        desc0.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc0.location.id = resident_device_id;
        desc0.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDescriptors.push_back(desc0);
    }
    
    // 应用权限 (此时硬件 P2P 链路自动打通)
    CHECK_CU(cuMemSetAccess(d_ptr_base, phy_size, accessDescriptors.data(), accessDescriptors.size()));

    // ----------------------------------------------------------------
    // [核心逻辑] 计算真正的 Tensor 指针
    // ----------------------------------------------------------------
    // d_ptr_base 大块初始段
    // final_ptr  计算offset (小 Tensor 真正locate的地方)
    void* final_ptr = (void*)((uintptr_t)d_ptr_base + offset);

    // ----------------------------------------------------------------
    // [PyTorch Binding] 封装
    // ----------------------------------------------------------------
    auto options = torch::TensorOptions()
                    .device(torch::kCUDA, resident_device_id) // 户籍地：Rank 0
                    .dtype(dtype);
                    
    // 使用计算后的 final_ptr 创建 Tensor
    torch::Tensor t = torch::from_blob(final_ptr, shape, nop_deleter, options);
    
    return t;
}

// ------------------------------------------------------------------
// PyBind 定义
// ------------------------------------------------------------------
PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("import_tensor_from_fd", &import_tensor_from_fd, 
          "Import VMM Tensor from FD with Offset (Pure Driver API)");
}