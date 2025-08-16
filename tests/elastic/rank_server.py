# rank_server_full.py
from socketserver import ThreadingTCPServer, StreamRequestHandler
from threading import Thread
from collections import defaultdict
import socket, os

# --- Server state ---
_counts = defaultdict(int)
_global = 0

# --- Request handler ---
class H(StreamRequestHandler):
    def handle(self):
        global _global
        line = self.rfile.readline().strip().decode()
        
        if line.startswith("RELEASE_LAST"):
            # Handle rank release - decrement global counter
            if _global > 0:
                _global -= 1
                self.wfile.write("OK\n".encode())
            else:
                self.wfile.write("ERROR: No ranks to release\n".encode())
        else:
            # Handle rank assignment (original behavior)
            host = line
            local = _counts[host]
            _counts[host] += 1
            remote = _global
            _global += 1
            self.wfile.write(f"{local} {remote}\n".encode())

# --- TCPServer subclass to reuse port immediately ---
class ReusableTCPServer(ThreadingTCPServer):
    allow_reuse_address = True

# --- Lazy-start server ---
def start_server(port=9999):
    try:
        server = ReusableTCPServer(("0.0.0.0", port), H)
        Thread(target=server.serve_forever, daemon=True).start()
    except OSError:
        pass  # another process already started the server

# --- Client API ---
def get_rank(server="127.0.0.1", port=9999):
    s = socket.create_connection((server, port))
    s.sendall(f"{os.uname().nodename}\n".encode())
    return tuple(map(int, s.recv(1024).decode().split()))

def release_rank(server="127.0.0.1", port=9999):
    """Release a rank (decrement the global counter by 1)"""
    s = socket.create_connection((server, port))
    s.sendall("RELEASE_LAST\n".encode())
    response = s.recv(1024).decode().strip()
    s.close()
    return response == "OK"

# --- Example usage ---
if __name__ == "__main__":
    print("Getting rank:", get_rank())
    print("Releasing rank:", release_rank())
