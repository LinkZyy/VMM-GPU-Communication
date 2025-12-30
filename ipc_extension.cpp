#include <torch/extension.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <vector>
#include <iostream>

void nop_deleter(void* ptr) { }

#define CHECK_CU(x) { \
    CUresult res = x; \
    if (res != CUDA_SUCCESS) { \
        const char* err; \
        cuGetErrorString(res, &err); \
        std::cerr << "[VMM_IPC_ERROR] API Call Failed at line " << __LINE__ << std::endl; \
        std::cerr << "[VMM_IPC_ERROR] Code: " << res << " String: " << err << std::endl; \
        throw std::runtime_error(std::string("CUDA Error: ") + err); \
    } \
}

// 辅助函数：开启 Peer Access (Rank 1 需要开启对 Rank 0 的访问)
void enable_peer_access(int target_device) {
    int current_device;
    cudaGetDevice(&current_device);
    if (current_device == target_device) return;

    int can_access = 0;
    cudaDeviceCanAccessPeer(&can_access, current_device, target_device);
    if (can_access) {
        // 尝试开启，如果已经开启会返回错误但我们可以忽略，或者检查错误码
        cudaError_t err = cudaDeviceEnablePeerAccess(target_device, 0);
        if (err != cudaSuccess && err != cudaErrorPeerAccessAlreadyEnabled) {
             std::cerr << "[VMM_IPC] Warning: Failed to enable Peer Access: " << cudaGetErrorString(err) << std::endl;
        }
    }
}

torch::Tensor import_tensor_from_fd(int fd, int64_t size, 
                                    int mapping_device_id,  // Rank 1
                                    int resident_device_id, // Rank 0
                                    std::vector<int64_t> shape, 
                                    c10::ScalarType dtype) {
    
    cudaSetDevice(mapping_device_id);
    enable_peer_access(resident_device_id);

    // 1. Import
    CUmemGenericAllocationHandle handle;
    CHECK_CU(cuMemImportFromShareableHandle(&handle, 
                                            (void*)(uintptr_t)fd, 
                                            CU_MEM_HANDLE_TYPE_POSIX_FILE_DESCRIPTOR));

    // 2. Reserve
    CUdeviceptr d_ptr;
    CHECK_CU(cuMemAddressReserve(&d_ptr, size, 0, 0, 0));

    // 3. Map
    CHECK_CU(cuMemMap(d_ptr, size, 0, handle, 0));

    // 4. Set Access (关键修改！！！)
    // 我们需要构建一个 AccessDesc 数组，同时允许 Mapping Device (1) 和 Resident Device (0) 访问
    // 这样无论 PyTorch 决定用哪个 Device 的 Stream 去搬运数据，都不会 Permission Denied
    
    std::vector<CUmemAccessDesc> accessDescriptors;
    
    // 给当前设备 (Rank 1) 权限 -> 用于 P2P 计算
    CUmemAccessDesc desc1 = {};
    desc1.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    desc1.location.id = mapping_device_id;
    desc1.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    accessDescriptors.push_back(desc1);

    // 给源设备 (Rank 0) 权限 -> 用于 PyTorch 内部调度 (因为 tensor 标记为 cuda:0)
    if (mapping_device_id != resident_device_id) {
        CUmemAccessDesc desc0 = {};
        desc0.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        desc0.location.id = resident_device_id;
        desc0.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
        accessDescriptors.push_back(desc0);
    }
    
    std::cout << "[VMM_IPC] Setting access for " << accessDescriptors.size() << " devices..." << std::endl;
    CHECK_CU(cuMemSetAccess(d_ptr, size, accessDescriptors.data(), accessDescriptors.size()));

    // 5. Create Tensor (标记为 Resident Device 0 以通过检查)
    auto options = torch::TensorOptions()
                    .device(torch::kCUDA, resident_device_id)
                    .dtype(dtype);
                    
    torch::Tensor t = torch::from_blob((void*)d_ptr, shape, nop_deleter, options);
    
    return t;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("import_tensor_from_fd", &import_tensor_from_fd, "Import VMM Tensor from FD");
}