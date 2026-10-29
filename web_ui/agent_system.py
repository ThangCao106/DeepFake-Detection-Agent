"""
Multi-Agent Deepfake Detection System

This module implements a multi-agent system for detecting deepfakes in images.
It uses:
1. Detection Agent: Runs ML model for deepfake detection
2. Analysis Agent: LLM-powered agent for explaining results
3. Orchestrator: Coordinates the workflow

Key Concepts Implemented (Kaggle Requirements):
- Multi-agent system (Sequential agents)
- Agent powered by LLM (Gemini)
- Custom tools (Deepfake detection tool)
- Sessions & Memory (Conversation history)
- Observability (Logging and tracing)
"""

import os
import sys
import json

# Safe encoding for Windows console (avoid charmap crash on emojis)
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, field

import warnings
warnings.filterwarnings("ignore", category=FutureWarning)

import google.generativeai as genai

try:
    from colorama import Fore, Style, init
    init(autoreset=True)
except ImportError:
    class _ColorDummy:
        def __getattr__(self, name):
            return ""
    Fore = _ColorDummy()
    Style = _ColorDummy()
    def init(*args, **kwargs):
        pass

from deepfake_detector import DeepfakeDetector
from config import Config

# Configure logging for observability
logging.basicConfig(
    level=getattr(logging, Config.LOG_LEVEL),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('agent_system.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


import re
import requests

@dataclass
class Message:
    """Represents a message in the conversation."""
    role: str  # 'user', 'agent', 'system'
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class ObservabilityMemoryStore:
    """
    Persistent memory store for Observability Agent & Multi-Modal Forensic Audits.
    Stores forensic decision records and certificates indexed by Certificate ID.
    """
    _instance = None

    def __init__(self, memory_file="dataset_pipeline/observability_memory.json"):
        self.memory_file = memory_file
        os.makedirs(os.path.dirname(memory_file), exist_ok=True)
        self.certificates: Dict[str, Dict[str, Any]] = {}
        self.chat_history: List[Dict[str, Any]] = []
        self._load_memory()

    @classmethod
    def get_instance(cls):
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance

    def _load_memory(self):
        if os.path.exists(self.memory_file):
            try:
                with open(self.memory_file, "r", encoding="utf-8") as f:
                    data = json.load(f)
                    self.certificates = data.get("certificates", {})
                    self.chat_history = data.get("chat_history", [])
                logger.info(f"Loaded {len(self.certificates)} forensic certificates into ObservabilityMemoryStore")
            except Exception as e:
                logger.error(f"Error loading ObservabilityMemoryStore: {e}")

    def save_certificate(self, cert_id: str, detection_result: Dict[str, Any]):
        """Store forensic analysis certificate by Certificate ID."""
        if not cert_id:
            return
        cert_id_upper = cert_id.strip().upper()
        self.certificates[cert_id_upper] = {
            "timestamp": datetime.now().isoformat(),
            "detection_result": detection_result
        }
        self._persist()

    def lookup_certificate(self, cert_id: str) -> Optional[Dict[str, Any]]:
        """Retrieve forensic analysis certificate by Certificate ID."""
        if not cert_id:
            return None
        cert_id_upper = cert_id.strip().upper()
        record = self.certificates.get(cert_id_upper)
        if record:
            return record.get("detection_result")
        
        # Fuzzy match search for certificate ID substring
        for cid, data in self.certificates.items():
            if cert_id_upper in cid:
                return data.get("detection_result")
        return None

    def add_chat_log(self, role: str, content: str, cert_id: str = None):
        """Add chat entry to persistent Observability Memory."""
        self.chat_history.append({
            "timestamp": datetime.now().isoformat(),
            "role": role,
            "content": content,
            "certificate_id": cert_id
        })
        # Keep last 100 entries
        if len(self.chat_history) > 100:
            self.chat_history = self.chat_history[-100:]
        self._persist()

    def _persist(self):
        try:
            with open(self.memory_file, "w", encoding="utf-8") as f:
                json.dump({
                    "certificates": self.certificates,
                    "chat_history": self.chat_history
                }, f, indent=2, default=str)
        except Exception as e:
            logger.error(f"Error persisting ObservabilityMemoryStore: {e}")


class ConversationMemory:
    """
    Manages conversation history and session state.
    Implements: Sessions & Memory concept from Kaggle requirements.
    """
    
    def __init__(self, max_history: int = getattr(Config, "MAX_CONVERSATION_HISTORY", 20)):
        self.messages: List[Message] = []
        self.max_history = max_history
        self.session_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.obs_memory = ObservabilityMemoryStore.get_instance()
        logger.info(f"Initialized conversation memory for session: {self.session_id}")
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to conversation history and sync with Observability Memory."""
        msg = Message(role=role, content=content, metadata=metadata or {})
        self.messages.append(msg)
        
        if len(self.messages) > self.max_history:
            self.messages = self.messages[-self.max_history:]
        
        cert_id = None
        if metadata and "forensic_certificate" in metadata:
            cert_id = metadata["forensic_certificate"].get("certificate_id")
            self.obs_memory.save_certificate(cert_id, metadata)
        
        self.obs_memory.add_chat_log(role, content, cert_id=cert_id)
        logger.debug(f"Added {role} message to memory and ObservabilityStore")
    
    def get_context(self) -> str:
        """Get formatted conversation context for LLM."""
        context_parts = []
        for msg in self.messages[-5:]:
            context_parts.append(f"{msg.role.upper()}: {msg.content}")
        return "\n".join(context_parts)
    
    def save_session(self, filepath: str = None):
        """Save conversation history to file."""
        if filepath is None:
            filepath = f"session_{self.session_id}.json"
        
        data = {
            "session_id": self.session_id,
            "messages": [
                {
                    "role": msg.role,
                    "content": msg.content,
                    "timestamp": msg.timestamp,
                    "metadata": msg.metadata
                }
                for msg in self.messages
            ]
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved session to {filepath}")


class DeepfakeDetectionTool:
    """
    Custom tool for deepfake detection.
    Implements: Custom tools concept from Kaggle requirements.
    """
    
    def __init__(self):
        self.detector = DeepfakeDetector()
        logger.info("Initialized DeepfakeDetectionTool")
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Run deepfake detection on media file (Image, Video, Audio).
        """
        logger.info(f"Running detection on: {image_path}")
        
        start_time = datetime.now()
        result = self.detector.predict_media(image_path)
        
        duration = (datetime.now() - start_time).total_seconds()
        result['processing_time'] = duration
        
        # Save to Observability Memory Store
        cert = result.get("forensic_certificate", {})
        cert_id = cert.get("certificate_id")
        if cert_id:
            ObservabilityMemoryStore.get_instance().save_certificate(cert_id, result)
        
        return result
    
    def get_description(self) -> str:
        """Get tool description for LLM context."""
        return """
        DeepfakeDetectionTool: Analyzes images, videos, and audio clips to detect if they are real or AI-generated deepfakes.
        Returns: label (Realism/Deepfake), confidence score, detailed probability scores, and Auditable Forensic Certificate.
        """


class AnalysisAgent:
    """
    LLM-powered agent for analyzing detection results, checking Certificate IDs, and answering queries.
    Connects to both Gemini and Claude via Observability Agent.
    """
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        genai.configure(api_key=api_key)
        self.model_candidates = ["gemini-3.6-flash", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash", "gemini-1.5-flash"]
        self.active_model_name = self._sanitize_model_name(Config.GEMINI_MODEL_NAME)
        self.model = genai.GenerativeModel(self.active_model_name)
        self.obs_memory = ObservabilityMemoryStore.get_instance()
        logger.info(f"Initialized AnalysisAgent with primary model {self.active_model_name}")

    def _sanitize_model_name(self, name: str) -> str:
        """Remove 'models/' prefix if present to ensure API compatibility."""
        if name.startswith("models/"):
            return name.replace("models/", "")
        return name
    
    def _call_claude_llm(self, prompt: str) -> Optional[str]:
        """Invoke Claude LLM via logged-in Observability Agent service (Port 8787)."""
        obs_key = os.getenv("API_KEY", "observability-agent-key")
        headers = {"X-Api-Key": obs_key, "Content-Type": "application/json"}
        
        # 1. Try local Observability Agent /api/chat endpoint (uses logged-in Claude CLI session)
        try:
            url = f"{Config.OBSERVABILITY_AGENT_URL}/api/chat"
            payload = {"session_id": "web-ui-session", "message": prompt}
            res = requests.post(url, headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                text = data.get("response") or data.get("content") or data.get("message")
                if text:
                    logger.info("Generated explanation via Observability Agent (Logged-in Claude)")
                    return text
        except Exception as e_obs:
            logger.debug(f"Observability Agent /api/chat endpoint failed: {e_obs}")

        # 2. Try OpenAI-compatible /v1/chat/completions endpoint
        try:
            url = f"{Config.OBSERVABILITY_AGENT_URL}/v1/chat/completions"
            payload = {
                "messages": [{"role": "user", "content": prompt}]
            }
            res = requests.post(url, headers=headers, json=payload, timeout=12)
            if res.status_code == 200:
                data = res.json()
                choices = data.get("choices", [])
                if choices:
                    text = choices[0].get("message", {}).get("content")
                    if text:
                        logger.info("Generated explanation via Observability Agent OpenAI-Compat (Logged-in Claude)")
                        return text
        except Exception as e_v1:
            logger.debug(f"Observability Agent /v1/chat/completions endpoint failed: {e_v1}")

        return None

    def analyze(self, detection_result: Dict[str, Any], user_query: str = None, 
                conversation_context: str = "", language: str = 'en') -> str:
        """
        Analyze detection results or lookup Certificate ID and provide authoritative LLM explanation.
        """
        logger.info(f"AnalysisAgent processing request (language: {language})")
        
        # Check if user query specifies a Certificate ID to check (CERT-DF, CERT-AUD, CERT-VID, or CERT-YYYYMMDD)
        cert_id_match = re.search(r'(CERT-(?:DF|AUD|VID|\d{8})-[A-Z0-9]+)', user_query or '', re.IGNORECASE)
        if cert_id_match:
            target_cert_id = cert_id_match.group(1).upper()
            logger.info(f"Certificate ID query detected: {target_cert_id}")
            stored_result = self.obs_memory.lookup_certificate(target_cert_id)
            if stored_result:
                detection_result = stored_result
                user_query = f"Provide detailed forensic verification for Certificate ID: {target_cert_id}. Question: {user_query}"
            else:
                if language == 'vi':
                    return f"🔍 Không tìm thấy Certificate ID '{target_cert_id}' trong Observability Memory Store. Vui lòng đảm bảo hình ảnh đã được phân tích."
                return f"🔍 Certificate ID '{target_cert_id}' not found in Observability Memory Store. Please ensure the media has been analyzed."

        prompt = self._build_prompt(detection_result, user_query, conversation_context, language=language)
        
        # Try Claude first if configured
        claude_response = self._call_claude_llm(prompt)
        if claude_response:
            return claude_response

        # Fallback to Gemini with model name retry logic (1.5-flash -> 2.0-flash -> 1.5-pro)
        for model_name in [self.active_model_name] + self.model_candidates:
            try:
                model_inst = genai.GenerativeModel(model_name)
                response = model_inst.generate_content(prompt)
                analysis = response.text
                logger.info(f"AnalysisAgent completed analysis using Gemini ({model_name})")
                return analysis
            except Exception as e:
                logger.warning(f"Gemini model '{model_name}' failed: {e}. Trying next model candidate...")
        
        return f"Analytic Report: Media classified as '{detection_result.get('label')}' with {detection_result.get('confidence', 0):.1%} confidence. Certificate ID: {detection_result.get('forensic_certificate', {}).get('certificate_id', 'N/A')}."
    
    def _build_prompt(self, detection_result: Dict[str, Any], 
                     user_query: str = None, context: str = "", language: str = 'en') -> str:
        """Build the prompt for the LLM based on Auditable Forensic Decision Record."""
        
        cert = detection_result.get("forensic_certificate", {})
        evidence = cert.get("multi_dimensional_evidence", {})

        lang_instruction = (
            "IMPORTANT LANGUAGE REQUIREMENT: You MUST write your ENTIRE response in VIETNAMESE (Tiếng Việt). Use precise, professional Vietnamese forensic terminology."
            if language == 'vi'
            else "IMPORTANT LANGUAGE REQUIREMENT: You MUST write your ENTIRE response in ENGLISH. Use precise, professional forensic terminology."
        )
        
        base_prompt = f"""You are an Expert AI Media Forensics Investigator certifying synthetic media authenticity under EU AI Act Article 50 & DSA Auditable Media Standards.
{lang_instruction}

AUDITABLE DIGITAL FORENSIC DECISION RECORD:
- Certificate ID: {cert.get('certificate_id', 'N/A')}
- Timestamp: {cert.get('timestamp', 'N/A')}
- Analysis Mode: {cert.get('analysis_mode', 'SINGLE_STREAM_GENERAL')}
- Composite Risk Score: {cert.get('composite_risk_score', 0):.2%}
- Auditable Verdict: {cert.get('auditable_verdict', 'UNKNOWN')}

DUAL-STREAM MULTI-MODAL EVIDENCE BREAKDOWN:
{json.dumps(cert.get('dual_stream_breakdown', {}), indent=2) if cert.get('dual_stream_breakdown') else "Single stream general vision analysis."}

FOUR FORENSIC EVIDENCE DIMENSIONS:
1. Passive Neural Classifier:
   - Prediction Label: {detection_result.get('label', 'Unknown')}
   - Model Confidence: {detection_result.get('confidence', 0):.2%}
   - Class Probabilities: {json.dumps(evidence.get('passive_model_score', {}).get('class_probabilities', {}), indent=2)}

2. Frequency Domain Spectral Analysis (FFT 2D):
   - Spectral Anomaly Score: {evidence.get('spectral_frequency_analysis', {}).get('high_frequency_anomaly_score', 'N/A')}
   - Interpretation: {evidence.get('spectral_frequency_analysis', {}).get('interpretation', 'N/A')}

3. Spatial Noise & Variance Analysis:
   - Spatial Inconsistency Score: {evidence.get('spatial_inconsistency_analysis', {}).get('spatial_inconsistency_score', 'N/A')}
   - Interpretation: {evidence.get('spatial_inconsistency_analysis', {}).get('interpretation', 'N/A')}

4. Classifier Uncertainty & Entropy:
   - Entropy Score: {evidence.get('model_uncertainty_entropy', {}).get('entropy', 'N/A')}
   - Certainty Rating: {evidence.get('model_uncertainty_entropy', {}).get('interpretation', 'N/A')}

PREVIOUS CONVERSATION CONTEXT:
{context}

YOUR TASK: """
        
        if user_query:
            base_prompt += f"Answer the user's specific query: '{user_query}' using the Auditable Forensic Certificate evidence provided above."
        else:
            base_prompt += """Provide a concise Executive Verdict & Certificate Summary in clear natural language (2-3 short paragraphs max).
Explain key forensic findings (whether passive neural model, FFT 2D spectral anomaly, or spatial noise inconsistency detected synthetic artifacts or authentic patterns). Do not generate repetitive markdown tables, as key metrics are formatted separately."""
        
        return base_prompt



def safe_print(msg: str):
    """Prints message safely even on legacy Windows code pages without crashing."""
    try:
        print(msg)
    except Exception:
        try:
            clean = msg.encode('ascii', errors='replace').decode('ascii')
            print(clean)
        except Exception:
            pass


class DeepfakeDetectionOrchestrator:
    """
    Orchestrator agent that coordinates the multi-agent workflow.
    Implements: Multi-agent system (Sequential agents) from Kaggle requirements.
    """
    
    def __init__(self, gemini_api_key: str):
        # Initialize components
        self.detection_tool = DeepfakeDetectionTool()
        self.analysis_agent = AnalysisAgent(gemini_api_key)
        self.memory = ConversationMemory()
        
        logger.info("Initialized DeepfakeDetectionOrchestrator")
        safe_print(f"{Fore.GREEN}Deepfake Detection Agent System Initialized{Style.RESET_ALL}")
        safe_print(f"{Fore.CYAN}Session ID: {self.memory.session_id}{Style.RESET_ALL}\n")
    
    def process_image(self, image_path: str, user_query: str = None, language: str = 'en') -> str:
        """
        Process an image through the multi-agent workflow.
        
        Workflow:
        1. User provides image (and optional question)
        2. Detection Agent runs ML model
        3. Analysis Agent (LLM) explains results
        4. Response returned to user
        5. Conversation stored in memory
        
        Args:
            image_path: Path to image to analyze
            user_query: Optional question from user
            language: Language preference ('en' or 'vi')
            
        Returns:
            Analysis and explanation
        """
        logger.info(f"Processing image: {image_path} (language: {language})")
        
        # Add user message to memory with appropriate media modality
        ext = os.path.splitext(image_path)[1].lower() if image_path else ""
        if ext in ['.wav', '.mp3', '.flac', '.ogg', '.aac', '.m4a', '.wma']:
            media_modality_name = "audio"
        elif ext in ['.mp4', '.avi', '.mov', '.mkv', '.webm', '.flv']:
            media_modality_name = "video"
        else:
            media_modality_name = "image"
        
        user_msg = f"Analyze {media_modality_name}: {image_path}"
        if user_query:
            user_msg += f" | Question: {user_query}"
        self.memory.add_message("user", user_msg)
        
        # Step 1: Detection Agent - Run ML model
        safe_print(f"{Fore.YELLOW}Detection Agent: Analyzing {media_modality_name}...{Style.RESET_ALL}")
        detection_result = self.detection_tool.detect(image_path)
        
        if "error" in detection_result:
            error_msg = f"Error: {detection_result['error']}"
            self.memory.add_message("system", error_msg)
            return error_msg
        
        # Log detection results
        self.memory.add_message(
            "system", 
            f"Detection complete: {detection_result['label']} ({detection_result['confidence']:.2%})",
            metadata=detection_result
        )
        
        safe_print(f"{Fore.CYAN}Result: {detection_result['label']} "
                   f"(Confidence: {detection_result['confidence']:.2%}){Style.RESET_ALL}")
        
        # Step 2: Analysis Agent - LLM explains results
        safe_print(f"{Fore.YELLOW}Analysis Agent: Generating explanation...{Style.RESET_ALL}")
        
        conversation_context = self.memory.get_context()
        analysis = self.analysis_agent.analyze(
            detection_result, 
            user_query, 
            conversation_context,
            language=language
        )
        
        # Add analysis to memory
        self.memory.add_message("agent", analysis, metadata=detection_result)
        
        return analysis
    
    def ask_question(self, question: str, language: str = 'en') -> str:
        """
        Ask a follow-up question about previous detections.
        Uses conversation memory for context.
        """
        logger.info(f"User question: {question} (language: {language})")
        self.memory.add_message("user", question)
        
        print(f"{Fore.YELLOW}🧠 Analysis Agent: Processing question...{Style.RESET_ALL}")
        
        # Get last detection result from memory
        last_detection = None
        for msg in reversed(self.memory.messages):
            if msg.role == "system" and msg.metadata:
                last_detection = msg.metadata
                break
        
        if not last_detection:
            response = (
                "Chưa tìm thấy kết quả phân tích nào trước đó. Vui lòng tải lên và phân tích hình ảnh trước."
                if language == 'vi'
                else "No previous detection results found. Please analyze an image first."
            )
        else:
            context = self.memory.get_context()
            response = self.analysis_agent.analyze(last_detection, question, context, language=language)
        
        self.memory.add_message("agent", response)
        return response
    
    def save_session(self):
        """Save the current session."""
        self.memory.save_session()
        print(f"{Fore.GREEN}💾 Session saved{Style.RESET_ALL}")


def main():
    """Main function to run the agent system interactively."""
    
    # Validate configuration
    try:
        Config.validate()
    except ValueError as e:
        print(f"{Fore.RED}❌ Configuration Error: {e}{Style.RESET_ALL}")
        return
    
    # Initialize orchestrator
    orchestrator = DeepfakeDetectionOrchestrator(Config.GEMINI_API_KEY)
    
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Welcome to the Deepfake Detection Agent System!{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'='*60}{Style.RESET_ALL}\n")
    
    print("Commands:")
    print("  analyze <image_path> - Analyze an image for deepfakes")
    print("  ask <question> - Ask a question about previous results")
    print("  save - Save the current session")
    print("  quit - Exit the system\n")
    
    while True:
        try:
            user_input = input(f"{Fore.GREEN}You: {Style.RESET_ALL}").strip()
            
            if not user_input:
                continue
            
            if user_input.lower() == 'quit':
                orchestrator.save_session()
                print(f"\n{Fore.CYAN}Goodbye!{Style.RESET_ALL}")
                break
            
            elif user_input.lower() == 'save':
                orchestrator.save_session()
            
            elif user_input.lower().startswith('analyze '):
                image_path = user_input[8:].strip()
                print()
                response = orchestrator.process_image(image_path)
                print(f"\n{Fore.GREEN}Agent: {Style.RESET_ALL}{response}\n")
            
            elif user_input.lower().startswith('ask '):
                question = user_input[4:].strip()
                print()
                response = orchestrator.ask_question(question)
                print(f"\n{Fore.GREEN}Agent: {Style.RESET_ALL}{response}\n")
            
            else:
                print(f"{Fore.RED}Unknown command. Use: analyze, ask, save, or quit{Style.RESET_ALL}\n")
        
        except KeyboardInterrupt:
            print(f"\n\n{Fore.CYAN}Interrupted. Saving session...{Style.RESET_ALL}")
            orchestrator.save_session()
            break
        except Exception as e:
            logger.error(f"Error: {e}", exc_info=True)
            print(f"{Fore.RED}Error: {e}{Style.RESET_ALL}\n")


if __name__ == "__main__":
    main()
