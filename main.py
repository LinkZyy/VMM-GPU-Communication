# LD_PRELOAD=./libvmm_hook.so torchrun --nproc_per_node=2 main.py

import torch
import os
import time
from ipc_utils import VMMCommunicator, torch_mysend, torch_myrecv

# 必须通过 torchrun 启动
def main():
    rank = int(os.environ["LOCAL_RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    torch.cuda.set_device(rank)
    
    # 初始化通信器
    comm = VMMCommunicator(rank, world_size)
    
    if rank == 0:
        # --- Sender (Card A) ---
        print(f"[Rank 0] Initializing...")
        comm.connect_to(1)
        
        # 1. 分配 Tensor (触发 LD_PRELOAD Hook)
        # 注意：这里的大小必须足够大以触发 cudaMalloc (比如 >20MB)
        # 还要确保和 Hook 代码里写的 size 匹配
        num_elements = 32 * 1024 * 1024 
        x = torch.zeros(num_elements, device='cuda', dtype=torch.float32)
        x.fill_(3.14159) # 写入数据
        
        print(f"[Rank 0] Tensor allocated at {x.data_ptr()}")
        
        # 2. 发送
        torch_mysend(x, 1, comm)
        
        # 等待验证
        time.sleep(5)
        
        # 验证 Rank 1 是否修改了数据
        print(f"[Rank 0] Value after IPC: {x[0].item()}")
        
    elif rank == 1:
        # --- Receiver (Card B) ---
        print(f"[Rank 1] Waiting for connection...")
        comm.wait_for_connection()
        
        # 1. 接收
        y = torch_myrecv(0, comm)
        
        print(f"[Rank 1] Received Tensor at {y.data_ptr()}")
        print(f"[Rank 1] Content check: {y[0].item()}")
        
        # 2. 修改数据 (Zero Copy 验证)
        y.fill_(999.0)
        print(f"[Rank 1] Modified content to 999.0")
        
        time.sleep(5)

if __name__ == "__main__":
    main()