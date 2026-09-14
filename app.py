import os
import sys

# Add 1_web_ui directory to Python import path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "1_web_ui"))

from app import app, socketio

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
