import socket
import pickle
import struct
import argparse
import numpy as np
import threading
from queue import Queue
import time
import matplotlib.pyplot as plt
import pandas as pd


def send_msg(conn, obj):
    """Envia um objeto Python com pickle + cabeçalho de tamanho (4 bytes)."""
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))
    conn.sendall(header + data)


def recv_msg(conn):
    """Recebe um objeto Python usando cabeçalho de 4 bytes com o tamanho."""
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
    """Thread worker: envia submatriz A_sub e B para um servidor e recebe C_sub."""
    host, port = server_addr
    try:
        with socket.create_connection((host, port), timeout=6000) as s:
            send_msg(s, {'A_sub': A_sub.tolist(), 'B': B.tolist()})
            resp = recv_msg(s)

            if "error" in resp:
                raise RuntimeError(resp["error"])

            C_sub = np.array(resp["C_sub"])
            out_queue.put((idx, C_sub))
    except Exception as e:
        out_queue.put((idx, e))


def split_matrix_rows(A, parts):
    """
    Divide a matriz A em 'parts' blocos de linhas aproximadamente iguais.
    Garante que 'parts' nunca seja maior que o número de linhas.
    """
    n_rows = A.shape[0]
    parts = min(parts, n_rows)  # nunca mais workers que linhas

    sizes = [n_rows // parts] * parts
    for i in range(n_rows % parts):
        sizes[i] += 1

    blocks = []
    start = 0
    for sz in sizes:
        blocks.append(A[start:start + sz, :])
        start += sz

    return blocks


def distributed_matmul(A, B, servers):
    """
    Multiplicação distribuída:
    - Divide A por linhas
    - Distribui os blocos entre os servidores
    - Cada servidor calcula C_sub = A_sub x B
    - O cliente empilha (vstack) os resultados
    """
    num_workers = len(servers)
    parts = split_matrix_rows(A, num_workers)

    out_queue = Queue()
    threads = []

    for idx, part in enumerate(parts):
        server = servers[idx % len(servers)]
        t = threading.Thread(
            target=worker_send_receive,
            args=(server, part, B, out_queue, idx),
            daemon=True
        )
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

    C = np.vstack(results)
    return C


def naive_matmul(A, B):
    """
    Multiplicação LOCAL.
    Mesmo algoritmo conceitual usado no servidor (para ser comparável).
    """
    A = np.array(A, dtype=int)
    B = np.array(B, dtype=int)

    m, n = A.shape
    nB, p = B.shape
    if n != nB:
        raise ValueError(f"Dimensões incompatíveis: A={A.shape}, B={B.shape}")

    C = np.zeros((m, p), dtype=int)

    for i in range(m):
        for k in range(n):
            aik = A[i, k]
            for j in range(p):
                C[i, j] += aik * B[k, j]

    return C


def plot_matrix_numbers(M, title="Matriz"):
    """
    Mostra a matriz M como tabela de números (até 15x15) em janela gráfica.
    Serve para visualizar A, B e C de forma mais organizada.
    """
    M = np.array(M)

    max_rows = min(20, M.shape[0])
    max_cols = min(20, M.shape[1])
    sub = M[:max_rows, :max_cols]

    fig, ax = plt.subplots(figsize=(max_cols * 0.7, max_rows * 0.5))
    ax.set_axis_off()

    table = ax.table(cellText=sub, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(8)
    table.scale(1.2, 1.2)

    ax.set_title(f"{title} (mostrando {max_rows}x{max_cols})", pad=10)
    plt.tight_layout()
    plt.show()


def ler_dimensoes_usuario():
    """
    Lê as dimensões de A e B garantindo que A(m×n) x B(n×p) seja válida.
    """
    while True:
        try:
            print("\n=== Definição das dimensões das matrizes ===")
            m = int(input("Número de LINHAS de A: "))
            n = int(input("Número de COLUNAS de A: "))
            nB = int(input("Número de LINHAS de B: "))
            p = int(input("Número de COLUNAS de B: "))

            if m <= 0 or n <= 0 or nB <= 0 or p <= 0:
                print("\n[ERRO] Todas as dimensões devem ser > 0.\n")
                continue

            if n != nB:
                print("\n[ERRO] Para multiplicar: A(m×n) × B(n×p).")
                print("      As COLUNAS de A devem ser iguais às LINHAS de B.\n")
                continue

            return m, n, p
        except ValueError:
            print("\n[ERRO] Digite apenas inteiros.\n")


def run_benchmark(servers, seed=42):

    # Agora vai até 2000, com maior densidade no início e mais espaçado depois
    sizes = [20, 40, 60, 80, 120, 160, 200, 240, 280, 320,
             360, 400, 450, 500, 600, 700, 800, 900, 1000,
             1200, 1400, 1600, 1800, 2000]

    results = []
    np.random.seed(seed)

    for n in sizes:
        print(f"\n>>> Testando tamanho {n}x{n} ...")

        A = np.random.randint(-5, 6, size=(n, n))
        B = np.random.randint(-5, 6, size=(n, n))

        # --- Local ---
        t0 = time.perf_counter()
        C_local = naive_matmul(A, B)
        t1 = time.perf_counter()
        t_local = t1 - t0

        # --- Distribuído ---
        t0 = time.perf_counter()
        C_dist = distributed_matmul(A, B, servers)
        t1 = time.perf_counter()
        t_dist = t1 - t0

        # --- Correctness ---
        C_np = A @ B
        correct_local = np.array_equal(C_local, C_np)
        correct_dist = np.array_equal(C_dist, C_np)

        results.append({
            "size": n,
            "local_time": t_local,
            "dist_time": t_dist,
            "local_ok": correct_local,
            "dist_ok": correct_dist
        })

    # --- DataFrame ---
    df = pd.DataFrame(results)
    print("\n\n=========== RESULTADOS DO BENCHMARK ===========\n")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.6f}"))
    print("\n================================================\n")

    # --- Plot ---
    plt.figure(figsize=(10,5))
    plt.plot(df["size"], df["local_time"], marker="o", label="Execução Local")
    plt.plot(df["size"], df["dist_time"], marker="o", label="Execução Distribuída")
    plt.xlabel("Tamanho da matriz (N x N)")
    plt.ylabel("Tempo (s)")
    plt.title("Comparação de desempenho: Execução Local vs Execução Distribuída")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--servers', nargs='+', required=True,
        help="Lista de servidores host:port (ex: 127.0.0.1:5001 127.0.0.1:5002)"
    )
    parser.add_argument(
        '--seed', type=int, default=42,
        help="Seed para geração aleatória das matrizes"
    )
    parser.add_argument(
        '--benchmark', action='store_true',
        help="Executa benchmark automático com 15 tamanhos"
    )

    args = parser.parse_args()

    servers = [(host, int(port)) for host, port in
               (s.split(":") for s in args.servers)]

    # === MODO BENCHMARK ===
    if args.benchmark:
        run_benchmark(servers, seed=args.seed)
        exit(0)

    # === MODO NORMAL (seu comportamento original) ===
    m, n, p = ler_dimensoes_usuario()
    np.random.seed(args.seed)
    A = np.random.randint(-5, 6, size=(m, n))
    B = np.random.randint(-5, 6, size=(n, p))