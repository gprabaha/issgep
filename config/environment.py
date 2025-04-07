# config/environment.py
import os
import socket
import getpass

def detect_runtime_environment() -> dict:
    """Auto-detect runtime environment based on hostname or filesystem."""
    hostname = socket.gethostname().lower()
    username = getpass.getuser()

    if "grace" in hostname or os.path.exists("/gpfs/gibbs/"):
        return {"is_cluster": True, "is_grace": True, "prabaha_local": False}
    elif "milgram" in hostname or os.path.exists("/gpfs/milgram/"):
        return {"is_cluster": True, "is_grace": False, "prabaha_local": False}
    elif username == "prabaha":
        return {"is_cluster": False, "is_grace": False, "prabaha_local": True}
    else:
        return {"is_cluster": False, "is_grace": False, "prabaha_local": False}
