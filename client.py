import socket
import pickle
import struct
import argparse
import numpy as np
import threading
from queue import Queue

def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))
    conn.sendall(header + data)

def recv_msg(conn):
    header = b''
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("server closed")
        header += chunk
    msglen = struct.unpack('!I', header)[0]
    data = b''
    while len(data) < msglen:
        chunk = conn.recv(min(4096, msglen - len(data)))
        if not chunk:
            raise ConnectionError("server closed during data")
        data += chunk
    return pickle.loads(data)

def worker_send_receive(server_addr, A_sub, B, out_queue, idx):
    host, port = server_addr
    try:
        with socket.create_connection((host, port), timeout=10) as s:
            send_msg(s, {'A_sub': A_sub.tolist(), 'B': B.tolist()})
            resp = recv_msg(s)
            if 'error' in resp:
                raise RuntimeError(resp['error'])
            C_sub = np.array(resp['C_sub'])
            out_queue.put((idx, C_sub))
    except Exception as e:
        out_queue.put((idx, e))

def split_matrix_rows(A, parts):
    n_rows = A.shape[0]
    sizes = [n_rows // parts] * parts
    for i in range(n_rows % parts):
        sizes[i] += 1
    res = []
    start = 0
    for sz in sizes:
        res.append(A[start:start+sz, :])
        start += sz
    return res

def distributed_matmul(A, B, servers):
    num_workers = len(servers)
    parts = split_matrix_rows(A, num_workers)
    out_queue = Queue()
    threads = []
    for i, part in enumerate(parts):
        # server selection: if fewer servers than parts, reuse servers circularly
        server = servers[i % len(servers)]
        t = threading.Thread(target=worker_send_receive, args=(server, part, B, out_queue, i), daemon=True)
        threads.append(t)
        t.start()

    results = [None] * len(parts)
    for _ in range(len(parts)):
        idx, payload = out_queue.get()
        if isinstance(payload, Exception):
            raise RuntimeError(f"Worker {idx} failed: {payload}")
        results[idx] = payload

    for t in threads:
        t.join(timeout=0.1)

    # concatenate results in order
    C = np.vstack(results) if len(results) > 0 else np.array([[]])
    return C

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--servers', nargs='+', required=True,
                        help='List of server addresses host:port')
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    servers = []
    for s in args.servers:
        host, port = s.split(':')
        servers.append((host, int(port)))

    # Example: generate matrices A (m x n) and B (n x p)
    np.random.seed(args.seed)
    m, n, p = 6, 3, 4  # example sizes; adjust to test
    A = np.random.randint(-5, 6, size=(m, n))
    B = np.random.randint(-5, 6, size=(n, p))

    print("A =")
    print(A)
    print("\nB =")
    print(B)

    C = distributed_matmul(A, B, servers)

    print("\nC (distributed result) =")
    print(C)

    # verify with local multiplication
    C_expected = A @ B
    print("\nC (expected, local) =")
    print(C_expected)

    print("\nCheck equal:", np.array_equal(C, C_expected))
