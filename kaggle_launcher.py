"""
Kaggle/Colab launcher for app.py -- paste this into the notebook cell AFTER the
`%%writefile app.py` cell.

Why it lives here and not in app.py: Streamlit re-executes app.py on every
widget interaction and every browser session, so starting a tunnel from inside
it opens duplicate tunnels. The notebook process runs once, so it owns both the
Streamlit subprocess and the ngrok tunnel.
"""

import subprocess
import time

import requests
from pyngrok import ngrok

NGROK_AUTHTOKEN =
PORT = 8501


def launch():
    # Re-running the cell leaves the old server holding the port, which is what
    # "Port 8501 is not available" means. Clear both sides before starting.
    ngrok.kill()
    subprocess.run(["pkill", "-f", "streamlit run"], check=False)
    time.sleep(2)

    log = open("streamlit.log", "w")
    proc = subprocess.Popen(
        [
            "streamlit", "run", "app.py",
            f"--server.port={PORT}",
            "--server.headless=true",
            "--server.address=0.0.0.0",
            "--server.enableCORS=false",
            "--server.enableXsrfProtection=false",
            "--browser.gatherUsageStats=false",
        ],
        stdout=log,
        stderr=subprocess.STDOUT,
    )

    for _ in range(120):
        if proc.poll() is not None:
            raise RuntimeError(
                "Streamlit exited early:\n" + open("streamlit.log").read()
            )
        try:
            requests.get(f"http://localhost:{PORT}", timeout=2)
            break
        except requests.exceptions.RequestException:
            time.sleep(1)
    else:
        raise RuntimeError(
            "Streamlit never came up:\n" + open("streamlit.log").read()
        )

    ngrok.set_auth_token(NGROK_AUTHTOKEN)
    url = ngrok.connect(addr=PORT, proto="http").public_url.replace("http://", "https://", 1)
    print("Streamlit is up.")
    print("Public URL:", url)
    return proc, url


if __name__ == "__main__":
    proc, url = launch()
    # Keep the notebook cell alive -- the tunnel dies when this process exits.
    proc.wait()
