import torch
import vmm_ipc
import socket
import os
import struct
import pickle
import multiprocessing.reduction
import time  # 记得引入 time

def get_socket_path(rank):
    return f"/tmp/vmm_ipc_socket_{rank}"

class VMMCommunicator:
    def __init__(self, rank, world_size):
        self.rank = rank
        self.world_size = world_size
        self.sock_path = get_socket_path(rank)
        
        # Receiver 初始化监听
        if rank == 1: 
            # 清理旧的 socket 文件
            if os.path.exists(self.sock_path):
                try:
                    os.remove(self.sock_path)
                except OSError:
                    pass # 忽略删除错误
            
            self.server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            self.server.bind(self.sock_path)
            self.server.listen(1)
            print(f"[Rank {rank}] Listening on {self.sock_path}")
    
    def wait_for_connection(self):
        self.conn, _ = self.server.accept()
        print(f"[Rank {self.rank}] Accepted connection")

    def connect_to(self, target_rank, retry_count=10):
        # Sender 连接 Receiver
        target_path = get_socket_path(target_rank)
        self.conn = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        
        for i in range(retry_count):
            try:
                print(f"[Rank {self.rank}] Trying to connect to {target_path} (Attempt {i+1}/{retry_count})...")
                self.conn.connect(target_path)
                print(f"[Rank {self.rank}] Connected to Rank {target_rank}")
                return # 连接成功，直接返回
            except (FileNotFoundError, ConnectionRefusedError) as e:
                # 文件还没生成，或者生成了但对方还没开始 listen
                time.sleep(1)
                continue
        
        # 如果重试多次失败，抛出异常
        raise RuntimeError(f"[Rank {self.rank}] Failed to connect to Rank {target_rank} after retries")

# ---------------- API 实现 (保持不变) ----------------
def torch_mysend(tensor, dest_rank, comm: VMMCommunicator):
    # 获取实际 FD (从临时文件读取)
    fd_file = f"/tmp/vmm_fd_rank_{comm.rank}"
    
    # 增加一个简单的等待，确保 FD 文件已生成
    while not os.path.exists(fd_file):
        time.sleep(0.1)

    with open(fd_file, 'r') as f:
        fd_str = f.read().strip()
        actual_fd = int(fd_str)
    phy_size_bytes = 128 * 1024 * 1024
    
    metadata = {
        'shape': list(tensor.shape),
        'dtype': tensor.dtype,
        'size': tensor.element_size() * tensor.numel(),
        'phy_size': phy_size_bytes # 134217728
    }
    
    meta_bytes = pickle.dumps(metadata)
    comm.conn.send(struct.pack("I", len(meta_bytes)))
    comm.conn.send(meta_bytes)
    multiprocessing.reduction.send_handle(comm.conn, actual_fd, os.getpid())
    print(f"[MySend] Sent Tensor {tensor.shape} with FD {actual_fd}")

def torch_myrecv(src_rank, comm: VMMCommunicator):
    # 1. 接收元数据长度
    len_bytes = comm.conn.recv(4)
    if not len_bytes:
        raise RuntimeError("Connection closed unexpectedly")
    meta_len = struct.unpack("I", len_bytes)[0]
    
    # 2. 接收元数据
    meta_bytes = comm.conn.recv(meta_len)
    metadata = pickle.loads(meta_bytes)
    
    # 3. 接收 FD
    print(f"[MyRecv] Waiting for FD...")
    fd = multiprocessing.reduction.recv_handle(comm.conn)
    print(f"[MyRecv] Got Metadata: {metadata}, FD: {fd}")
    
    # 获取物理大小
    alloc_size = metadata.get('phy_size', 1024 * 1024 * 128) 
    
    # 参数：src_rank (即 resident_device_id)
    new_tensor = vmm_ipc.import_tensor_from_fd(
        fd, 
        alloc_size, 
        torch.cuda.current_device(), # 参数3: mapping_device_id (当前 Rank 1)
        src_rank,                    # 参数4: resident_device_id (数据源 Rank 0)k
        metadata['shape'],           # 参数5
        metadata['dtype']            # 参数6
    )
    
    return new_tensor