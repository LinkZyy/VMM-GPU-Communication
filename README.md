# PyTorch Zero-Copy IPC via CUDA VMM (Proof of Concept)

This project demonstrates a high-performance, **Zero-Copy** Inter-Process Communication (IPC) mechanism for PyTorch tensors across multiple GPUs on the same node.

By leveraging **CUDA Virtual Memory Management (VMM) APIs** and **Linux Unix Domain Sockets**, this implementation bypasses standard memory copies. Instead, it allows different processes to map their virtual address spaces to the **same physical GPU memory page**, achieving bandwidths close to hardware limits (e.g., NVLink speeds).

## 🚀 Key Features

* **True Zero-Copy:** No `cudaMemcpy` involved. Data is accessed directly via P2P (Peer-to-Peer) mappings.
* **Non-Invasive:** Uses `LD_PRELOAD` to intercept `cudaMalloc`, requiring zero modifications to PyTorch source code.

## 📂 Project Structure

### 1. Core System (The "Hijacker")
* **`vmm_hook.cpp`**
    * **Role:** The interceptor library.
    * **Function:** Hooks into the `cudaMalloc` symbol. Instead of standard allocation, it uses `cuMemCreate` to allocate shareable physical memory.
    * **Key Logic:** Exports the memory handle to a Linux File Descriptor (FD) and persists it temporarily (via `/tmp`) for IPC. It also configures access permissions to allow all GPUs to read/write this memory (crucial for P2P).

### 2. PyTorch Extension (The "Reconstructor")
* **`ipc_extension.cpp`**
    * **Role:** C++ backend for the receiver process.
    * **Function:** Takes a received FD and maps it into the current process's virtual address space using `cuMemMap`.
    * **Key Logic:** Wraps the raw mapped pointer into a `torch::Tensor` using `torch::from_blob`, ensuring the correct device residency is set to satisfy PyTorch's internal checks.
* **`setup.py`**
    * **Role:** Build script for compiling `ipc_extension.cpp` into a Python module (`vmm_ipc`).

### 3. Communication Layer (The "Courier")
* **`ipc_utils.py`**
    * **Role:** Handles metadata and FD transmission.
    * **Function:** Uses Unix Domain Sockets to send tensor metadata (shape, dtype, size) and uses `SCM_RIGHTS` (via `multiprocessing.reduction`) to pass the valid Kernel FD to another process.

### 4. Application Entry (The "Test")
* **`main.py / main2.py`**

---

## ⚡ Quick Start

### Step 1: Compile the VMM Hook
Compile the hook into a shared library (`.so`). This requires `nvcc`.

```bash
g++ -shared -fPIC -o libvmm_hook.so vmm_hook.cpp -I/usr/local/cuda/include -lcuda -ldl
```

### Step 2: Install the Python Extension
```bash
python setup.py install
```

### Step 3: Run the tests
```bash
LD_PRELOAD=./libvmm_hook.so torchrun --nproc_per_node=2 main2.py
```