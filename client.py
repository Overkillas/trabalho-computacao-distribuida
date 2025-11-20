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


# Envia um objeto para o servidor.
# Aqui a ideia é transformar o objeto em bytes com pickle,
# colocar um cabeçalho dizendo o tamanho total e mandar tudo pela rede.
def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))
    conn.sendall(header + data)


# Recebe um objeto enviado pelo servidor.
# Primeiro chega o tamanho (4 bytes), depois buscamos o restante dos dados.
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


# Esta função é usada pelas threads.
# Cada thread envia seu pedaço da matriz A e a matriz B inteira para um servidor.
# Depois ela recebe a parte correspondente do resultado.
def worker_send_receive(server_addr, A_sub, B, out_queue, idx):
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


# Divide a matriz A em blocos de linhas para distribuir entre os servidores.
# Cada servidor fica responsável por calcular um pedaço da multiplicação.
def split_matrix_rows(A, parts):
    n_rows = A.shape[0]
    parts = min(parts, n_rows)

    sizes = [n_rows // parts] * parts
    for i in range(n_rows % parts):
        sizes[i] += 1

    blocks = []
    start = 0
    for sz in sizes:
        blocks.append(A[start:start + sz, :])
        start += sz

    return blocks


# Aqui acontece a multiplicação distribuída de verdade.
# O cliente separa A em blocos, manda cada bloco para um servidor,
# recebe os resultados e monta a matriz final empilhando tudo.
def distributed_matmul(A, B, servers):
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

    return np.vstack(results)


# Esta é a multiplicação local usando 3 loops.
# É a forma “padrão” ensinada na disciplina, então usamos ela
# para comparar com a versão distribuída.
def naive_matmul(A, B):
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


# Apenas gera uma visualização bonitinha de matriz até 20x20.
# Isso ajuda muito para analisar exemplos pequenos.
def plot_matrix_numbers(M, title="Matriz"):
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


# Apenas pergunta ao usuário o tamanho de A e B e garante que é possível multiplicar.
def ler_dimensoes_usuario():
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


######################################################
# BENCHMARK — roda vários tamanhos automaticamente
# para comparar o local vs distribuído.
######################################################
def run_benchmark(servers, seed=42):

    sizes = [20, 40, 60, 80, 120, 160, 200, 240, 280, 320,
             360, 400, 450, 500, 600, 700, 800, 900, 1000,
             1200, 1400, 1600, 1800, 2000]

    results = []
    np.random.seed(seed)

    for n in sizes:
        print(f"\n>>> Testando tamanho {n}x{n} ...")

        A = np.random.randint(-5, 6, size=(n, n))
        B = np.random.randint(-5, 6, size=(n, n))

        # Tempo local
        t0 = time.perf_counter()
        C_local = naive_matmul(A, B)
        t1 = time.perf_counter()
        t_local = t1 - t0

        # Tempo distribuído
        t0 = time.perf_counter()
        C_dist = distributed_matmul(A, B, servers)
        t1 = time.perf_counter()
        t_dist = t1 - t0

        # Validação
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

    df = pd.DataFrame(results)
    print("\n\n=========== RESULTADOS DO BENCHMARK ===========\n")
    print(df.to_string(index=False,
                       float_format=lambda x: f"{x:.6f}"))
    print("\n================================================\n")

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


######################################################
# PARTE PRINCIPAL
######################################################
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
        help="Executa benchmark automático com vários tamanhos"
    )

    args = parser.parse_args()

    servers = [(host, int(port)) for host, port in
               (s.split(":") for s in args.servers)]

    # Se o usuário pediu benchmark:
    if args.benchmark:
        run_benchmark(servers, seed=args.seed)
        exit(0)

    # Modo normal
    m, n, p = ler_dimensoes_usuario()
    np.random.seed(args.seed)
    A = np.random.randint(-5, 6, size=(m, n))
    B = np.random.randint(-5, 6, size=(n, p))
