import socket
import threading
import pickle
import struct
import argparse
import numpy as np

# helpers: send/recv with length-prefix (4 bytes, network order)
def send_msg(conn, obj):
    data = pickle.dumps(obj, protocol=pickle.HIGHEST_PROTOCOL)
    header = struct.pack('!I', len(data))
    conn.sendall(header + data)

def recv_msg(conn):
    # read 4 bytes length
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

def handle_client(conn, addr):
    try:
        payload = recv_msg(conn)
        # payload expected: dict with keys 'A_sub' and 'B'
        A_sub = np.array(payload['A_sub'])
        B = np.array(payload['B'])
        # compute
        C_sub = A_sub @ B
        # optionally send back metadata
        send_msg(conn, {'C_sub': C_sub.tolist()})
    except Exception as e:
        try:
            send_msg(conn, {'error': str(e)})
        except Exception:
            pass
    finally:
        conn.close()

def start_server(host, port):
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    s.bind((host, port))
    s.listen()
    print(f"[SERVER] Listening on {host}:{port}")
    try:
        while True:
            conn, addr = s.accept()
            print(f"[SERVER] Connection from {addr}")
            t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
            t.start()
    except KeyboardInterrupt:
        print("\n[SERVER] Shutting down")
    finally:
        s.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--host', default='127.0.0.1')
    parser.add_argument('--port', type=int, default=5001)
    args = parser.parse_args()
    start_server(args.host, args.port)
