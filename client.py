import socket
import pickle
import struct
import argparse
import numpy as np
import threading
from queue import Queue
import time  # para medir tempos


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
        with socket.create_connection((host, port), timeout=60) as s:
            # Envia submatriz de A e matriz B
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
    """
    Divide A em blocos de linhas e distribui entre os servidores.
    Garante que NUNCA haverá mais workers do que linhas de A,
    para evitar blocos vazios (0xN) causando erros estranhos.
    """
    if A.shape[0] == 0:
        raise ValueError("Matriz A não pode ter zero linhas.")

    # limite de workers = min(nº de servidores, nº de linhas de A)
    num_workers = min(len(servers), A.shape[0])

    parts = split_matrix_rows(A, num_workers)
    out_queue = Queue()
    threads = []

    for i, part in enumerate(parts):
        server = servers[i % len(servers)]
        t = threading.Thread(
            target=worker_send_receive,
            args=(server, part, B, out_queue, i),
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

    C = np.vstack(results) if len(results) > 0 else np.array([[]])
    return C


def naive_matmul(A, B):
    """
    Multiplicação de matrizes ingênua em Python puro (3 laços for).
    Serve como 'versão local sequencial' para comparação com a distribuída.
    """
    m, n = A.shape
    nB, p = B.shape
    assert n == nB
    C = np.zeros((m, p), dtype=int)

    for i in range(m):         # linhas de A
        for k in range(n):     # colunas de A / linhas de B
            aik = A[i, k]
            for j in range(p): # colunas de B
                C[i, j] += aik * B[k, j]
    return C


def ler_dimensoes_usuario():
    """
    Lê as dimensões das matrizes A e B do usuário.
    Garante que as matrizes possam ser multiplicadas:
    A (m x n)  e  B (n x p)
    """
    while True:
        try:
            print("\n=== Definição das dimensões das matrizes ===")
            m = int(input("Número de LINHAS da matriz A: "))
            n = int(input("Número de COLUNAS da matriz A: "))

            nB = int(input("Número de LINHAS da matriz B: "))
            p = int(input("Número de COLUNAS da matriz B: "))

            if m <= 0 or n <= 0 or nB <= 0 or p <= 0:
                print("\n[ERRO] Todas as dimensões devem ser maiores que zero. Tente novamente.\n")
                continue

            if n != nB:
                print("\n[ERRO] Não é possível multiplicar A ({}x{}) por B ({}x{}).".format(m, n, nB, p))
                print("       O número de COLUNAS de A deve ser igual ao número de LINHAS de B.")
                print("       Por favor, informe novas dimensões.\n")
                continue

            return m, n, p  # B terá (n x p)

        except ValueError:
            print("\n[ERRO] Digite apenas números inteiros. Tente novamente.\n")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--servers', nargs='+', required=True,
                        help='Lista de servidores no formato host:port')
    parser.add_argument('--seed', type=int, default=42,
                        help='Seed para geração aleatória das matrizes')
    args = parser.parse_args()

    # Monta lista de servidores
    servers = []
    for s in args.servers:
        host, port = s.split(':')
        servers.append((host, int(port)))

    # Lê dimensões desejadas do usuário, garantindo que a multiplicação seja possível
    m, n, p = ler_dimensoes_usuario()

    # Gera matrizes aleatórias com as dimensões informadas
    np.random.seed(args.seed)
    A = np.random.randint(-5, 6, size=(m, n))
    B = np.random.randint(-5, 6, size=(n, p))

    print("\n=== Matrizes geradas aleatoriamente ===")
    print("\nMatriz A ({}x{}):".format(m, n))
    print(A)
    print("\nMatriz B ({}x{}):".format(n, p))
    print(B)

    # 1) Multiplicação LOCAL INGUÊNUA (3 for)
    print("\nCalculando multiplicação LOCAL (ingênua, 3 laços for)...")
    t0_local = time.perf_counter()
    C_local = naive_matmul(A, B)
    t1_local = time.perf_counter()
    tempo_local = t1_local - t0_local
    print(f"Tempo da multiplicação LOCAL (ingênua): {tempo_local:.6f} segundos")

    # 2) Multiplicação DISTRIBUÍDA
    print("\nEnviando partes da matriz A para os servidores (cálculo DISTRIBUÍDO)...")
    t0_dist = time.perf_counter()
    C_dist = distributed_matmul(A, B, servers)
    t1_dist = time.perf_counter()
    tempo_dist = t1_dist - t0_dist

    print("\nMatriz C (resultado DISTRIBUÍDO) - dimensão {}x{}:".format(C_dist.shape[0], C_dist.shape[1]))
    print(C_dist)
    print(f"\nTempo da multiplicação DISTRIBUÍDA: {tempo_dist:.6f} segundos")

    # 3) Validação com NumPy (referência de corretude)
    print("\nValidando resultados com NumPy (A @ B)...")
    C_numpy = A @ B
    iguais_local = np.array_equal(C_local, C_numpy)
    iguais_dist = np.array_equal(C_dist, C_numpy)

    print("Local ingênuo == NumPy?     ", iguais_local)
    print("Distribuído    == NumPy?    ", iguais_dist)

    print(f"\nRESUMO DE TEMPOS:")
    print(f"  Local (ingênuo, 3 for): {tempo_local:.6f} s")
    print(f"  Distribuído (NumPy nos servidores): {tempo_dist:.6f} s")

    # 4) Comparação de desempenho LOCAL x DISTRIBUÍDO
    print("\n=== Comparação de desempenho LOCAL x DISTRIBUÍDO ===")
    if tempo_dist > 0:
        fator = tempo_local / tempo_dist  # quanto a distribuída é mais rápida (ou mais lenta)
        if tempo_dist < tempo_local:
            # distribuída mais rápida
            diff_pct = (tempo_local - tempo_dist) / tempo_local * 100
            print(f"A execução DISTRIBUÍDA foi aproximadamente {fator:.2f} vezes mais rápida que a LOCAL ingênua.")
            print(f"Isto representa uma redução de {diff_pct:.2f}% no tempo de execução.")
        else:
            # distribuída mais lenta (matrizes pequenas, overhead domina)
            fator_inverso = tempo_dist / tempo_local
            diff_pct = (tempo_dist - tempo_local) / tempo_local * 100
            print(f"A execução DISTRIBUÍDA foi aproximadamente {fator_inverso:.2f} vezes mais lenta que a LOCAL ingênua.")
            print(f"Isto representa um aumento de {diff_pct:.2f}% no tempo de execução.")

        print("\nObservação:")
        print("- Para matrizes muito pequenas, o overhead de comunicação pode tornar a versão distribuída mais lenta.")
        print("- À medida que o tamanho das matrizes cresce, o custo do cálculo domina e a abordagem distribuída tende")
        print("  a se tornar relativamente mais vantajosa em relação à versão local ingênua (3 laços).")
    else:
        print("Tempo local muito pequeno para calcular o fator de comparação.")
