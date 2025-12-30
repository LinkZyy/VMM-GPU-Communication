Technical Architecture Reference: PyTorch Zero-Copy IPC via CUDA VMM
1. Design Philosophy
This implementation strictly follows First Principles and Occam's Razor to achieve efficient cross-process memory sharing without modifying the PyTorch source code.

1.1 First Principles Approach
We deconstruct GPU memory management into its fundamental components:

Physical Memory: The actual storage resource on the GPU (Physical Handle).

Virtual Addressing: The process-specific view of that memory (Virtual Address).

Tensor Structure: A wrapper containing a pointer and metadata (Shape/Stride).

By decoupling physical allocation from virtual mapping, we achieve Zero-Copy IPC. Instead of copying data between buffers using cudaMemcpy, we establish a new virtual mapping in the receiver process that points to the exact same physical memory page allocated by the sender.

1.2 Occam's Razor (Implementation Strategy)
To minimize incidental complexity and avoid over-engineering:

Non-Invasive: We do not modify PyTorch's core C++ allocator or recompile the framework.

Interception: We use LD_PRELOAD to intercept cudaMalloc calls at the dynamic linking level. This allows us to transparently replace standard memory allocation with CUDA Virtual Memory Management (VMM) APIs only when necessary.

2. System Architecture
The system consists of three distinct modules that handle interception, transmission, and reconstruction.

Module I: The Interception Layer (libvmm_hook.so)
Role: Transparently replaces memory allocation logic.

Mechanism:

Intercept: Hooks into the cudaMalloc symbol triggered by PyTorch.

Allocate: Instead of standard allocation, calls cuMemCreate to allocate a shareable physical memory handle.

Export: Converts the physical handle into a Linux File Descriptor (FD) via cuMemExportToShareableHandle.

Persist: Temporarily stores the FD (e.g., in /tmp keyed by Rank/PID) for IPC retrieval.

Map: Maps the physical memory to the sender's virtual address space and returns the pointer to PyTorch.

Module II: The Transport Layer (ipc_utils.py)
Role: Handles the inter-process communication of metadata and handles.

Mechanism:

Metadata Transmission: Serializes Tensor attributes (Shape, Dtype, Stride, Physical Size) and sends them via a standard socket.

FD transmission: Uses the Unix Domain Socket SCM_RIGHTS mechanism (multiprocessing.reduction.send_handle). This is critical because FDs are process-local integers; the kernel must clone the underlying resource reference into the receiver's file descriptor table.

Module III: The Reconstruction Layer (ipc_extension.cpp)
Role: Reconstructs the Tensor in the receiver process without physical allocation.

Mechanism:

Import: Retrieves the physical allocation handle from the received FD via cuMemImportFromShareableHandle.

Reserve: Reserves a range of Virtual Address (VA) space in the receiver's context via cuMemAddressReserve.

Map: Maps the imported physical handle to the reserved VA range via cuMemMap.

Access Control: Explicitly grants read/write permissions to the mapped range.

Encapsulation: Wraps the raw pointer into a torch::Tensor using torch::from_blob, explicitly marking the device residency to match the source.

3. Data Flow Diagram
Plaintext

[Rank 0 (Sender)]                        [Kernel / Driver]                       [Rank 1 (Receiver)]
       |                                         |                                        |
1. torch.zeros(...)                              |                                        |
       |                                         |                                        |
2. Hook: cudaMalloc() -----------------> [Alloc Physical Page]                            |
       |                                 [Create Kernel Object]                           |
       |                                         |                                        |
3. torch_mysend()                                |                                        |
   (Send Metadata) ---------------------------------------------------------------------> |
   (Send FD via SCM_RIGHTS) ---------------------+                                        |
                                                 |                                4. torch_myrecv()
                                                 |                                        |
                                                 +----------------------------->  5. cuMemImport(FD)
                                                                                          |
                                         [Map Kernel Obj -> Virtual Addr] <---------------|
                                                                                          |
                                         [Update Page Table Permissions] <----------------|
                                                                                          |
       | <------------------------- (Direct P2P Access) ----------------------- 6. Access Tensor
4. Key Technical Specifications
4.1 Virtual Memory Management (VMM)
Unlike legacy cudaMalloc, the VMM API requires explicit management of the memory lifecycle:

Physical Allocation: cuMemCreate

VA Reservation: cuMemAddressReserve

Mapping: cuMemMap

Access Rights: cuMemSetAccess

4.2 Cross-Process Handle Sharing
Concept: A CUDA Handle is opaque and process-local. To share it, it must be exported to an OS-level File Descriptor (FD).

Implication: The FD represents a reference to the underlying kernel object (e.g., nvidia-fs or dma-buf). Ownership is shared; the physical memory is released only when all FDs are closed and all mappings are unmapped.

4.3 Dual-Device Access Permissions
A critical finding during implementation was the handling of access permissions in a multi-GPU context.

The Issue: When Rank 1 accesses a Tensor that physically resides on Rank 0, PyTorch may dispatch kernels on either device's stream.

The Solution: The cuMemSetAccess call must explicitly grant permissions to both devices:

mapping_device (Rank 1): To allow the receiver process to read/write.

resident_device (Rank 0): To allow the data owner (and PyTorch's internal consistency checks) to access the memory.

4.4 Device Residency Masquerading
The Problem: torch::from_blob usually assumes memory belongs to the current device.

The Solution: The receiver must initialize the Tensor with device=resident_device (Rank 0), even though the process is running on Rank 1. This accurately reflects the physical location of the data and enables Zero-Copy P2P access via NVLink/PCIe.

5. Future Optimization Path
While the current LD_PRELOAD approach serves as a functional Proof of Concept (PoC), production environments should adopt a Pluggable Allocator strategy:

Implementation: Implement the c10::cuda::CUDACachingAllocator::CUDAAllocator C++ interface.

Benefit: Allows granular control over memory pools, avoids hardcoded size alignments (e.g., the 128MB static alignment used in the PoC), and integrates natively with PyTorch's memory management system without relying on library injection.