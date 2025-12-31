# LD_PRELOAD=./libvmm_hook.so torchrun --nproc_per_node=2 caching-test.py

import torch
import os
import time
import struct
import pickle
import multiprocessing.reduction
from ipc_utils import VMMCommunicator

# 配置: 20GB 大块
HUGE_BLOCK_SIZE = 20 * 1024 * 1024 * 1024 
# 配置: 100MB 小块
SMALL_TENSOR_SIZE = 100 * 1024 * 1024

def run_sender(rank, comm):
    comm.connect_to(1)
    
    print(f"[Sender] 1. Allocating HUGE tensor ({HUGE_BLOCK_SIZE/1024**3} GB) to trigger VMM Hook...")
    # 这会触发 cudaMalloc，Hook 会记录 FD 和 BaseAddr
    huge_block = torch.empty(HUGE_BLOCK_SIZE, dtype=torch.int8, device='cuda')
    base_ptr = huge_block.data_ptr()
    
    # 读取 Hook 留下的信息
    time.sleep(0.5) # 等文件写入
    with open(f"/tmp/vmm_fd_rank_{rank}", "r") as f:
        content = f.read().strip().split()
        base_fd = int(content[0])
        hook_base_addr = int(content[1])
    
    print(f"[Sender] -> Hook reported: FD={base_fd}, Addr={hex(hook_base_addr)}")
    assert base_ptr == hook_base_addr, "Mismatch between PyTorch ptr and Hook ptr!"

    # 关键步骤：释放 huge_block，但 PyTorch 不会归还给 OS，而是放入缓存
    print(f"[Sender] 2. Deleting HUGE tensor (Release to PyTorch Cache)...")
    del huge_block 
    
    # 此时显存被 PyTorch 占着，但里面没数据
    
    print(f"[Sender] 3. Allocating SMALL tensors from the cached block...")
    # PyTorch 会复用刚才那 20GB 的空间
    t1 = torch.ones(SMALL_TENSOR_SIZE, dtype=torch.uint8, device='cuda') # 占前面 100MB
    t2 = torch.full((SMALL_TENSOR_SIZE,), 2, dtype=torch.uint8, device='cuda') # 占后面 100MB
    
    # 计算 t2 的偏移量
    t2_ptr = t2.data_ptr()
    offset = t2_ptr - base_ptr
    
    print(f"[Sender] -> t2 allocated at {hex(t2_ptr)}")
    print(f"[Sender] -> Offset from base: {offset} bytes ({offset/1024/1024} MB)")
    
    # 如果特殊情况下 PyTorch 决定重新申请，则测试终止！
    if offset < 0 or offset > HUGE_BLOCK_SIZE:
        print("\033[91m[WARNING] PyTorch did NOT reuse the block! Test invalid.\033[0m")
        exit(-1)
        

    # 发送 t2
    # 注意：我们要传的是 base_fd (大块的句柄) 和 offset (小块的位置)
    metadata = {
        'shape': list(t2.shape),
        'dtype': t2.dtype,
        'size': t2.numel() * t2.element_size(),
        'phy_size': HUGE_BLOCK_SIZE, # 映射时需要映射整个大块(或者足够大)
        'offset': offset             # 新增：偏移量
    }
    
    print(f"[Sender] 4. Sending t2 (FD={base_fd}, Offset={offset})...")
    
    # 发送逻辑
    meta_bytes = pickle.dumps(metadata)
    comm.conn.send(struct.pack("I", len(meta_bytes)))
    comm.conn.send(meta_bytes)
    multiprocessing.reduction.send_handle(comm.conn, base_fd, os.getpid())
    
    # 等待验证
    time.sleep(2)
    print(f"[Sender] t2[0] is now: {t2[0].item()}")
    if t2[0].item() == 100:
        print("\033[92m[SUCCESS] Remote modification verified!\033[0m")

def run_receiver(rank, comm):
    import vmm_ipc # 导入编译好的扩展
    comm.wait_for_connection()
    
    # 接收逻辑
    len_bytes = comm.conn.recv(4)
    meta_len = struct.unpack("I", len_bytes)[0]
    metadata = pickle.loads(comm.conn.recv(meta_len))
    fd = multiprocessing.reduction.recv_handle(comm.conn)
    
    print(f"[Receiver] Got metadata: {metadata}, FD: {fd}")
    
    # 调用 C++ 扩展 (带 offset)
    y = vmm_ipc.import_tensor_from_fd(
        fd,
        metadata['phy_size'],  # 映射整个 20GB 大小
        metadata['offset'],    # 偏移量
        torch.cuda.current_device(),
        0,                     # Source Rank
        metadata['shape'],
        metadata['dtype']
    )
    
    print(f"[Receiver] Reconstructed tensor. Value check: {y[0].item()} (Expected 2)")
    
    # 修改数据
    print(f"[Receiver] Modifying data to 100...")
    y.fill_(100)
    torch.cuda.synchronize()

def main():
    rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(rank)
    comm = VMMCommunicator(rank, 2)
    if rank == 0: run_sender(rank, comm)
    else: run_receiver(rank, comm)

if __name__ == "__main__":
    main()