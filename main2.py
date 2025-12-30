# LD_PRELOAD=./libvmm_hook.so torchrun --nproc_per_node=2 main2.py

import torch
import os
import time
import argparse
from ipc_utils import VMMCommunicator, torch_mysend, torch_myrecv

# ================= Configuration =================
# 必须与 vmm_hook.cpp 的对齐逻辑保持一致
# 32M float32 = 128MB 物理内存
NUM_ELEMENTS = 32 * 1024 * 1024 
PHY_SIZE_BYTES = NUM_ELEMENTS * 4 

def log(rank, msg):
    print(f"[{time.strftime('%H:%M:%S')}] [Rank {rank}] {msg}")

def check_p2p_support(current_device, target_device):
    """检查硬件 P2P 支持情况"""
    can_access = torch.cuda.can_device_access_peer(current_device, target_device)
    status = "\033[92mSUPPORTED\033[0m" if can_access else "\033[91mNOT SUPPORTED (Fallback via SysMem?)\033[0m"
    return status

def run_sender(rank, world_size, comm):
    target_rank = 1
    log(rank, f"=== Test Phase 1: Environment Setup ===")
    log(rank, f"Device: {torch.cuda.get_device_name(0)}")
    
    # 1. 连接接收端
    comm.connect_to(target_rank)
    
    # 2. 分配显存 (触发 LD_PRELOAD Hook)
    log(rank, "Allocating VMM Tensor...")
    # 填充一些特定 pattern 的数据
    x = torch.zeros(NUM_ELEMENTS, device='cuda', dtype=torch.float32)
    # 写入 Pattern: [0, 1, 2, ...]
    torch.arange(NUM_ELEMENTS, out=x) 
    
    log(rank, f"Tensor Allocated. Shape: {x.shape}, Physical Size: {PHY_SIZE_BYTES / 1024 / 1024:.2f} MB")
    log(rank, f"Data Preview (First 5): {x[:5].tolist()}")

    # 3. 发送句柄 (测量控制面延迟)
    log(rank, "=== Test Phase 2: Control Plane (IPC Handshake) ===")
    t0 = time.perf_counter()
    torch_mysend(x, target_rank, comm)
    t1 = time.perf_counter()
    log(rank, f"Metadata & FD Sent. Time cost: {(t1 - t0)*1000:.3f} ms")

    # 4. 等待接收端验证
    log(rank, "Waiting for receiver to modify data...")
    time.sleep(2) # 简单的同步，生产环境可用 barrier

    # 5. 验证 Zero-Copy (接收端修改后，发送端应该能看到变化)
    log(rank, "=== Test Phase 3: Zero-Copy Verification (Read-Back) ===")
    # 接收端应该把第一个元素改成了 -999
    val_check = x[0].item()
    if val_check == -999.0:
        log(rank, "\033[92m[PASS] Data modified by remote rank is visible locally!\033[0m")
    else:
        log(rank, f"\033[91m[FAIL] Expected -999.0, got {val_check}. Is Zero-Copy working?\033[0m")
    
    time.sleep(2) # 保持进程存活

def run_receiver(rank, world_size, comm):
    src_rank = 0
    log(rank, f"=== Test Phase 1: Environment Setup ===")
    # 检查是否能访问 Rank 0
    p2p_status = check_p2p_support(torch.cuda.current_device(), src_rank)
    log(rank, f"P2P Access to Rank {src_rank}: {p2p_status}")

    log(rank, "Waiting for connection...")
    comm.wait_for_connection()

    # 1. 接收 Tensor (测量映射耗时)
    log(rank, "=== Test Phase 2: Control Plane (Import & Map) ===")
    t0 = time.perf_counter()
    y = torch_myrecv(src_rank, comm)
    torch.cuda.synchronize() # 确保 mapping 完成
    t1 = time.perf_counter()
    
    log(rank, f"Tensor Reconstructed. Import & Map Latency: {(t1 - t0)*1000:.3f} ms")
    log(rank, f"Mapped Tensor Device: {y.device} (Should be cuda:{src_rank})")

    # 2. 数据完整性校验
    log(rank, "=== Test Phase 3: Data Integrity Check ===")
    # 检查 Pattern 是否为 [0, 1, 2, ...]
    # 抽样检查以节省时间
    head_check = y[:5].tolist()
    expected = [0.0, 1.0, 2.0, 3.0, 4.0]
    
    if head_check == expected:
         log(rank, "\033[92m[PASS] Data content matches sender!\033[0m")
    else:
         log(rank, f"\033[91m[FAIL] Data mismatch! Got {head_check}\033[0m")

    # 3. 写入测试 (In-place Modification)
    log(rank, "=== Test Phase 4: Write Latency & Zero-Copy Write ===")
    log(rank, "Modifying data (y[0] = -999.0)...")
    
    t_w0 = time.perf_counter()
    # 这是一个远程写操作 (通过 NVLink/PCIe 写回 Rank 0 显存)
    y[0] = -999.0 
    torch.cuda.synchronize()
    t_w1 = time.perf_counter()
    
    log(rank, f"Write completed. Single Element Access Latency: {(t_w1 - t_w0)*1000*1000:.3f} us")
    
    # 4. 带宽测试 (简单的 Memcpy)
    log(rank, "=== Test Phase 5: P2P Bandwidth Benchmark ===")
    # 创建一个本地 buffer
    local_buff = torch.empty_like(y, device=torch.cuda.current_device())
    
    # 预热
    local_buff.copy_(y)
    torch.cuda.synchronize()
    
    # 测速
    iterations = 10
    start_ev = torch.cuda.Event(enable_timing=True)
    end_ev = torch.cuda.Event(enable_timing=True)
    
    start_ev.record()
    for _ in range(iterations):
        local_buff.copy_(y) # P2P Read: Device 0 -> Device 1
    end_ev.record()
    torch.cuda.synchronize()
    
    avg_time_ms = start_ev.elapsed_time(end_ev) / iterations
    bandwidth_gbps = (PHY_SIZE_BYTES / (1024**3)) / (avg_time_ms / 1000)
    
    log(rank, f"P2P Read Bandwidth: \033[1;33m{bandwidth_gbps:.2f} GB/s\033[0m")
    
    time.sleep(2)

def main():
    # 获取环境变量
    rank = int(os.environ.get("LOCAL_RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    
    # 绑定设备
    torch.cuda.set_device(rank)
    
    # 初始化通信器
    comm = VMMCommunicator(rank, world_size)
    
    if rank == 0:
        run_sender(rank, world_size, comm)
    elif rank == 1:
        run_receiver(rank, world_size, comm)

if __name__ == "__main__":
    main()
