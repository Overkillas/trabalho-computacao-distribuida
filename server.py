import socket
import threading
import pickle
import struct
import argparse
import numpy as np

# Mesma lógica do cliente: envia objetos pela rede usando um cabeçalho
# que diz o tamanho do conteúdo.
def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))
    conn.sendall(header + data)


# Também igual ao cliente: primeiro recebemos o tamanho da mensagem,
# depois os dados completos.
def recv_msg(conn):
    header = b''
    while len(header) < 4:
        chunk = conn.recv(4 - len(header))
        if not chunk:
            raise ConnectionError("client closed")
        header += chunk

    msglen = struct.unpack('!I', header)[0]
    data = b''

    while len(data) < msglen:
        chunk = conn.recv(min(4096, msglen - len(data)))
        if not chunk:
            raise ConnectionError("client closed during data")
        data += chunk

    return pickle.loads(data)


# Multiplicação de matrizes simples com 3 loops.
# Essa lógica precisa ser a mesma do cliente para comparação.
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


# Cada cliente que conecta no servidor passa por aqui.
# O servidor recebe o pedaço da matriz, faz a multiplicação e devolve o resultado.
def handle_client(conn, addr):
    try:
        payload = recv_msg(conn)
        A_sub = payload['A_sub']
        B = payload['B']

        C_sub = naive_matmul(A_sub, B)

        send_msg(conn, {'C_sub': C_sub.tolist()})
    except Exception as e:
        try:
            send_msg(conn, {'error': str(e)})
        except:
            pass
    finally:
        conn.close()


# Inicia o servidor.
# Ele fica ouvindo novas conexões e cria uma thread para cada cliente.
def start_server(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen()
    print(f"[SERVER] Rodando em {host}:{port}")

    try:
        while True:
            conn, addr = s.accept()
            print(f"[SERVER] Conexão recebida de {addr}")
            threading.Thread(target=handle_client, args=(conn, addr), daemon=True).start()

    except KeyboardInterrupt:
        print("\nServidor finalizado.")
    finally:
        s.close()


# Parte principal do servidor
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=5001)
    args = parser.parse_args()

    start_server(args.host, args.port)
