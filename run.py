import subprocess
import sys

def run_fastapi():
    subprocess.run([sys.executable, "main.py"])

def run_streamlit():
    subprocess.run([sys.executable, "-m", "streamlit", "run", "streamlit_app.py"])

if __name__ == "__main__":
    import threading
    threading.Thread(target=run_fastapi).start()
    run_streamlit()