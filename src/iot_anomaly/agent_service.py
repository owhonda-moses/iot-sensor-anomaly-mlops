import subprocess
import os
from flask import Flask, jsonify

# start prefect agent as a background process to poll for new flow runs
subprocess.Popen(["prefect", "agent", "start", "-p", "default-agent-pool"])


app = Flask(__name__)

@app.route("/health", methods=["GET"])
def health():
    """A simple health check endpoint."""
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)