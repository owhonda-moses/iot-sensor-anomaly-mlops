import subprocess
import os
import sys
from flask import Flask, jsonify

# Start prefect agent
print("Starting Prefect agent in the background...")
subprocess.Popen(
    ["prefect", "agent", "start", "-p", "docker-pool"],
    stdout=sys.stdout, 
    stderr=sys.stderr,
)

# Start a simple web server for Cloud Run health checks
app = Flask(__name__)
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

if __name__ == "__main__":
    print("Starting health check server...")
    port = int(os.environ.get("PORT", 8080))
    app.run(host="0.0.0.0", port=port)