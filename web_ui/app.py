"""
Flask Web Application for Deepfake Detection Agent System

This web app provides a beautiful, interactive interface for the multi-agent
deepfake detection system with real-time communication and stunning UI.
"""

import os
import logging
from datetime import datetime
from flask import Flask, render_template, request, jsonify, send_from_directory
from flask_socketio import SocketIO, emit
from flask_cors import CORS
from werkzeug.utils import secure_filename

from agent_system import DeepfakeDetectionOrchestrator
from config import Config

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = 'deepfake-detection-secret-key-2024'
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 64 * 1024 * 1024  # 64MB max file size

# Allowed file extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp', 'gif', 'mp4', 'avi', 'mov', 'mkv', 'webm', 'mp3', 'wav'}

# Initialize extensions
CORS(app)
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='eventlet')

# Create upload folder if it doesn't exist
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global orchestrator instance (will be initialized when API key is available)
orchestrator = None


def allowed_file(filename):
    """Check if file extension is allowed."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def init_orchestrator():
    """Initialize the orchestrator if API key is available."""
    global orchestrator
    if orchestrator is None:
        try:
            if Config.GEMINI_API_KEY:
                orchestrator = DeepfakeDetectionOrchestrator(Config.GEMINI_API_KEY)
                logger.info("Orchestrator initialized successfully")
            else:
                logger.warning("GEMINI_API_KEY not set - LLM features disabled")
        except Exception as e:
            logger.error(f"Failed to initialize orchestrator: {e}")


# Routes
@app.route('/')
def index():
    """Serve the main web interface."""
    return render_template('index.html')


@app.route('/api/status')
def status():
    """Check API and agent status."""
    return jsonify({
        'status': 'online',
        'agent_ready': orchestrator is not None,
        'gemini_configured': bool(Config.GEMINI_API_KEY),
        'timestamp': datetime.now().isoformat()
    })


@app.route('/api/upload', methods=['POST'])
def upload_file():
    """Handle image upload."""
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type. Allowed: PNG, JPG, JPEG, WEBP, GIF'}), 400
    
    try:
        # Save file with secure filename
        filename = secure_filename(file.filename)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        filename = f"{timestamp}_{filename}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        logger.info(f"File uploaded: {filename}")
        
        return jsonify({
            'success': True,
            'filename': filename,
            'filepath': filepath
        })
    
    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500


@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)


@app.route('/api/verify/<cert_id>', methods=['GET'])
@app.route('/verify/<cert_id>', methods=['GET'])
def verify_certificate(cert_id):
    """Public API endpoint to query and verify Digital Forensic Certificates by ID."""
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from benchmark.deepfake_memory import get_forensic_certificate
        
        cert = get_forensic_certificate(cert_id)
        if cert.get("status") == "NOT_FOUND":
            return jsonify({
                "status": "NOT_FOUND",
                "valid": False,
                "message": f"No valid digital forensic certificate found for ID '{cert_id}'."
            }), 404
        
        return jsonify({
            "status": "VERIFIED_VALID",
            "valid": True,
            "certificate_id": cert.get("certificate_id"),
            "certificate": cert
        }), 200
    except Exception as e:
        logger.error(f"Certificate query error: {e}")
        return jsonify({"status": "ERROR", "message": str(e)}), 500


@app.route('/certificate/<cert_id>', methods=['GET'])
def view_certificate_page(cert_id):
    """Server-Side Rendered (SSR) Public Certificate Page with Schema.org ClaimReview JSON-LD for Googlebot."""
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from benchmark.deepfake_memory import get_forensic_certificate

        cert = get_forensic_certificate(cert_id)
        if cert.get("status") == "NOT_FOUND":
            return jsonify({"error": f"Certificate '{cert_id}' not found"}), 404

        template_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "observability", "static", "certificate.html"))
        if not os.path.exists(template_path):
            return verify_certificate(cert_id)

        with open(template_path, "r", encoding="utf-8") as f:
            html_content = f.read()

        verdict = cert.get("auditable_verdict", "VERIFIED")
        is_fake = "SUSPECTED" in verdict or "DETECTED" in verdict
        badge_class = "verdict-fake" if is_fake else "verdict-authentic"
        risk_color = "#dc2626" if is_fake else "#16a34a"

        risk_val = float(cert.get("composite_risk_score", 0.0))
        risk_percent = f"{risk_val * 100:.2f}"
        rating_val = f"{max(1, min(5, int((1.0 - risk_val) * 5)))}"

        ev = cert.get("multi_dimensional_evidence", {})
        sha256_hash = cert.get("media_integrity", {}).get("sha256") or cert.get("media_sha256") or "UNKNOWN_HASH"

        html_content = html_content.replace("{{CERTIFICATE_ID}}", str(cert.get("certificate_id", cert_id)))
        html_content = html_content.replace("{{VERDICT}}", verdict)
        html_content = html_content.replace("{{BADGE_CLASS}}", badge_class)
        html_content = html_content.replace("{{COMPOSITE_RISK_PERCENT}}", risk_percent)
        html_content = html_content.replace("{{RISK_COLOR}}", risk_color)
        html_content = html_content.replace("{{TIMESTAMP}}", str(cert.get("timestamp", "")))
        html_content = html_content.replace("{{TIMESTAMP_ISO}}", datetime.utcnow().isoformat() + "Z")
        html_content = html_content.replace("{{SHA256_HASH}}", str(sha256_hash))
        html_content = html_content.replace("{{RATING_VALUE}}", rating_val)

        pass_p = ev.get("passive_model_score", {})
        html_content = html_content.replace("{{PASSIVE_PRED}}", str(pass_p.get("prediction", "N/A")))
        html_content = html_content.replace("{{PASSIVE_CONF}}", f"{float(pass_p.get('confidence', 0.0))*100:.1f}%")

        spec_s = ev.get("spectral_frequency_analysis", {})
        html_content = html_content.replace("{{SPECTRAL_SCORE}}", str(spec_s.get("high_frequency_anomaly_score", "0.0")))

        spat_s = ev.get("spatial_inconsistency_analysis", {})
        html_content = html_content.replace("{{SPATIAL_SCORE}}", str(spat_s.get("spatial_inconsistency_score", "0.0")))

        ent_s = ev.get("model_uncertainty_entropy", {})
        html_content = html_content.replace("{{ENTROPY_SCORE}}", str(ent_s.get("entropy", "0.0")))

        html_content = html_content.replace("{{COMPLIANCE_STANDARD}}", str(cert.get("compliance_standard", "EU AI Act Article 50 & DSA Auditable Media")))

        return html_content, 200, {'Content-Type': 'text/html; charset=utf-8'}
    except Exception as e:
        logger.error(f"SSR cert error: {e}")
        return jsonify({"error": str(e)}), 500


@app.route('/sitemap.xml', methods=['GET'])
def get_sitemap_xml():
    """Generate dynamic XML sitemap listing all published certificates for Googlebot."""
    try:
        import sys
        sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
        from benchmark.deepfake_memory import get_all_certificates

        certs = get_all_certificates(limit=1000)
        xml_lines = [
            '<?xml version="1.0" encoding="UTF-8"?>',
            '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">'
        ]

        today_iso = datetime.now().strftime("%Y-%m-%d")
        xml_lines.append(f"  <url><loc>https://detection.dtpsoft.vn/</loc><lastmod>{today_iso}</lastmod><changefreq>daily</changefreq><priority>1.0</priority></url>")
        xml_lines.append(f"  <url><loc>https://detection.dtpsoft.vn/obsagent/</loc><lastmod>{today_iso}</lastmod><changefreq>daily</changefreq><priority>0.9</priority></url>")

        for c in certs:
            cid = c.get("certificate_id")
            created_at = (c.get("created_at") or today_iso)[:10]
            if cid:
                loc_url = f"https://detection.dtpsoft.vn/certificate/{cid}"
                xml_lines.append(f"  <url><loc>{loc_url}</loc><lastmod>{created_at}</lastmod><changefreq>weekly</changefreq><priority>0.8</priority></url>")

        xml_lines.append('</urlset>')
        xml_content = "\n".join(xml_lines)
        return xml_content, 200, {'Content-Type': 'application/xml; charset=utf-8'}
    except Exception as e:
        logger.error(f"Sitemap error: {e}")
        return jsonify({"error": str(e)}), 500


# WebSocket Events
@socketio.on('connect')
def handle_connect():
    """Handle client connection."""
    logger.info(f"Client connected: {request.sid}")
    emit('status', {
        'message': 'Connected to Deepfake Detection Agent System',
        'agent_ready': orchestrator is not None
    })


@socketio.on('disconnect')
def handle_disconnect():
    """Handle client disconnection."""
    logger.info(f"Client disconnected: {request.sid}")


@socketio.on('analyze_image')
def handle_analyze_image(data):
    """Handle image analysis request via WebSocket."""
    try:
        filepath = data.get('filepath')
        user_query = data.get('query', '')
        language = data.get('language', 'en')
        
        if not filepath or not os.path.exists(filepath):
            emit('error', {'message': 'Image file not found'})
            return
        
        # Emit status update
        emit('agent_status', {
            'agent': 'Detection Agent',
            'status': 'analyzing',
            'message': 'Analyzing image for deepfakes...'
        })
        
        # Initialize orchestrator if needed
        if orchestrator is None:
            init_orchestrator()
        
        if orchestrator is None:
            # Fallback to basic detection without LLM
            from deepfake_detector import DeepfakeDetector
            detector = DeepfakeDetector()
            
            emit('agent_status', {
                'agent': 'Detection Agent',
                'status': 'running',
                'message': 'Running ML model...'
            })
            
            result = detector.predict_media(filepath)
            
            if 'error' in result:
                emit('error', {'message': result['error']})
                return
            
            emit('detection_result', result)
            emit('agent_status', {
                'agent': 'Detection Agent',
                'status': 'complete',
                'message': 'Detection complete'
            })
            
            # Send basic explanation (no LLM)
            if language == 'vi':
                explanation = f"Tệp truyền thông được phân loại là '{result['label']}' với độ tin cậy {result['confidence']:.1%}. "
                if result['label'] == 'Realism':
                    explanation += "Cho thấy đây nhiều khả năng là hình ảnh thực tế, không phải tạo bằng AI."
                else:
                    explanation += "Cho thấy hình ảnh có dấu vết tạo bằng AI hoặc bị thao túng."
            else:
                explanation = f"The image was classified as '{result['label']}' with {result['confidence']:.1%} confidence. "
                if result['label'] == 'Realism':
                    explanation += "This suggests the image is likely a real photograph, not AI-generated."
                else:
                    explanation += "This suggests the image may be AI-generated or manipulated."
            
            emit('analysis_result', {'explanation': explanation})
        
        else:
            # Full agent workflow with LLM
            emit('agent_status', {
                'agent': 'Orchestrator',
                'status': 'coordinating',
                'message': 'Coordinating agents...'
            })
            
            # Process through orchestrator
            analysis = orchestrator.process_image(filepath, user_query, language=language)
            
            # Get the last detection result from memory
            detection_result = None
            for msg in reversed(orchestrator.memory.messages):
                if msg.role == 'system' and msg.metadata:
                    detection_result = msg.metadata
                    break
            
            if detection_result:
                emit('detection_result', detection_result)
            
            emit('agent_status', {
                'agent': 'Analysis Agent',
                'status': 'complete',
                'message': 'Analysis complete'
            })
            
            emit('analysis_result', {'explanation': analysis})
        
        logger.info(f"Analysis complete for: {filepath}")
    
    except Exception as e:
        logger.error(f"Analysis error: {e}", exc_info=True)
        emit('error', {'message': f'Analysis failed: {str(e)}'})


@socketio.on('ask_question')
def handle_ask_question(data):
    """Handle follow-up question via WebSocket."""
    try:
        question = data.get('question', '')
        language = data.get('language', 'en')
        
        if not question:
            emit('error', {'message': 'No question provided'})
            return
        
        if orchestrator is None:
            emit('error', {'message': 'Agent system not initialized. Please configure GEMINI_API_KEY.'})
            return
        
        emit('agent_status', {
            'agent': 'Analysis Agent',
            'status': 'thinking',
            'message': 'Processing your question...'
        })
        
        answer = orchestrator.ask_question(question, language=language)
        
        emit('agent_status', {
            'agent': 'Analysis Agent',
            'status': 'complete',
            'message': 'Answer ready'
        })
        
        emit('question_answer', {
            'question': question,
            'answer': answer
        })
        
        logger.info(f"Question answered: {question[:50]}...")
    
    except Exception as e:
        logger.error(f"Question error: {e}", exc_info=True)
        emit('error', {'message': f'Failed to answer question: {str(e)}'})


@app.route('/api/colab/callback', methods=['POST'])
def colab_callback():
    """Receive live tuning hyperparameters & evaluation metrics from Google Colab."""
    try:
        data = request.get_json(force=True) or {}
        logger.info(f"[colab_bridge] Received tuning callback from Colab: {data}")
        
        root_custom_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "custom_vit_model"))
        local_custom_dir = os.path.join(os.path.dirname(__file__), "custom_vit_model")
        custom_dir = root_custom_dir if os.path.exists(root_custom_dir) else local_custom_dir
        os.makedirs(custom_dir, exist_ok=True)
        
        # Ingest tuned calibration parameters
        calib_file = os.path.join(custom_dir, "calibration_params.json")
        calib_data = {
            "temperature_scaling_T": data.get("temperature_scaling_T", 1.6178),
            "optimal_threshold_tau": data.get("optimal_threshold_tau", 0.8118),
            "calibration_status": "CALIBRATED_FROM_COLAB_GPU",
            "accuracy": data.get("out_of_distribution_accuracy"),
            "dataset_trained": data.get("dataset_trained"),
            "timestamp": datetime.now().isoformat()
        }
        with open(calib_file, "w", encoding="utf-8") as f:
            import json
            json.dump(calib_data, f, indent=2)

        # Notify Web Dashboard UI via SocketIO
        socketio.emit('colab_sync_received', calib_data)

        # Reload orchestrator detector in memory
        global orchestrator
        if orchestrator and hasattr(orchestrator, "detector"):
            from deepfake_detector import DeepfakeDetector
            orchestrator.detector = DeepfakeDetector()
            logger.info("[colab_bridge] DeepfakeDetector reloaded with new Colab tuning parameters.")

        return jsonify({"status": "success", "message": "Colab tuning parameters ingested successfully", "calibrated": calib_data}), 200
    except Exception as e:
        logger.error(f"[colab_bridge] Callback error: {e}", exc_info=True)
        return jsonify({"status": "error", "message": str(e)}), 500


@socketio.on('save_session')
def handle_save_session():
    """Handle session save request."""
    try:
        if orchestrator is None:
            emit('error', {'message': 'No active session to save'})
            return
        
        orchestrator.save_session()
        
        emit('session_saved', {
            'message': 'Session saved successfully',
            'session_id': orchestrator.memory.session_id
        })
        
        logger.info(f"Session saved: {orchestrator.memory.session_id}")
    
    except Exception as e:
        logger.error(f"Save session error: {e}")
        emit('error', {'message': f'Failed to save session: {str(e)}'})


if __name__ == '__main__':
    # Initialize orchestrator on startup
    init_orchestrator()
    
    print("\n" + "="*60)
    print("🚀 Deepfake Detection Web App Starting...")
    print("="*60)
    print(f"📍 URL: http://localhost:5000")
    print(f"🤖 Agent System: {'Ready' if orchestrator else 'Limited (no API key)'}")
    print("="*60 + "\n")
    
    # Run the app
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
