import os
import sys
import warnings

# Suppress deprecated engine notices in console output
warnings.filterwarnings("ignore", message=".*Eventlet.*")
warnings.filterwarnings("ignore", message=".*google\\.generativeai.*")

# Add web_ui directory to Python import path
web_ui_dir = os.path.join(os.path.dirname(__file__), "web_ui")
if web_ui_dir not in sys.path:
    sys.path.insert(0, web_ui_dir)

import web_ui.app as web_app
app = web_app.app
socketio = web_app.socketio

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    socketio.run(app, host="0.0.0.0", port=port, debug=False)
