"""
ILORA RETREATS Concierge Bot with Google Gemini LLM
Optimized for high performance with background data refreshing.
Updated to use Gemini 1.5 Pro to solve token limit issues.
"""

import requests
import json
import logging
import os
import time
import threading
from typing import Optional, List, Dict, Any
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

# --- NEW: Import Google Generative AI ---
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# (keep other imports the same)
from vector_store import create_vector_store
from config import Config
from logger import setup_logger

logger = logging.getLogger("QAAgent")


class Welcome_IloraRetreatsConciergeBot:
    """
    Complete ILORA RETREATS Concierge Bot with intelligent guest/non-guest differentiation.
    **Uses Google Gemini 1.5 Pro** for high intelligence and massive context window.
    **Optimized with background data refreshing to ensure fast response times.**
    """
    def __init__(self):
        print("[DEBUG] Initializing WelcomeAIBot (Gemini Powered)...")
        start_init = time.time()

        self.config = Config

        # --- GEMINI CONFIGURATION ---
        # Replace with your actual Google API Key
        # You can get one here: https://aistudio.google.com/app/apikey
        self.llm_api_key = getattr(Config, "GOOGLE_API_KEY", os.environ.get("GOOGLE_API_KEY", "AIzaSyAoN1KBOVZ5gvBkp6dSnutUi1QTATSmcVw"))
        
        if not self.llm_api_key or "AIzaSyAoN1KBOVZ5gvBkp6dSnutUi1QTATSmcVw" in self.llm_api_key:
            logger.warning("⚠️ Google API Key is missing! Please set GOOGLE_API_KEY in Config or Environment.")

        # Configure the Gemini Client
        genai.configure(api_key=self.llm_api_key)
        
        # Using Gemini 1.5 Pro (Best for complex reasoning and handling large data)
        # It has a 1M+ token window, so it won't crash on large contexts.
        self.llm_model_name = "gemini-2.5-flash-lite" 

        # Sheet configuration
        self.sheet_api = getattr(Config, "GSHEET_WEBAPP_URL", None)
        self.qna_sheet = getattr(Config, "GSHEET_QNA_SHEET", "QnA_Manager")
        self.retriever_k = int(getattr(Config, "RETRIEVER_K", 5))
        self.sheet_refresh_interval = int(getattr(Config, "SHEET_REFRESH_INTERVAL", 300))
        self.sheet_fetch_timeout = float(getattr(Config, "SHEET_FETCH_TIMEOUT", 7.0))
        self.retrieve_timeout = float(getattr(Config, "RETRIEVER_TIMEOUT", 2.0))

        self.sheet_last_refresh = 0
        self.use_sheet = bool(self.sheet_api)
        self.http = requests.Session()

        # Data Storage
        self.qna_rows: List[Dict[str, Any]] = []

        # Chat History Management
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.chat_lock = threading.Lock()
        self.chat_history_limit = 15 # Increased limit because Gemini can handle it easily
        self.chat_history_persist = True
        self.chat_history_dir = os.path.join("data", "chat_histories")
        if self.chat_history_persist:
            os.makedirs(self.chat_history_dir, exist_ok=True)

        # Threading utilities
        self._executor = ThreadPoolExecutor(max_workers=4)
        self._stop_event = threading.Event() # For graceful shutdown

        # Load Initial Data and start background refreshing
        if self.use_sheet:
            try:
                # Initial blocking load to ensure agent has data at startup
                self._refresh_sheets(force=True)
                logger.info("Loaded Sheets data on init.")
                # Start the background refresh thread
                self._refresh_thread = threading.Thread(target=self._background_sheet_refresher, daemon=True)
                self._refresh_thread.start()
                logger.info("Started background sheet refresh thread.")
            except Exception as e:
                logger.warning(f"Sheets load failed: {e}")
                self.use_sheet = False

        logger.info(f"Welcome ConciergeBot ready with {self.llm_model_name}.")
        print(f"[DEBUG] Init complete in {time.time() - start_init:.2f}s")

    # ==================== DATA LOADING & BACKGROUND REFRESH ====================
    def _background_sheet_refresher(self):
        """Periodically refreshes sheet data in a background thread."""
        while not self._stop_event.is_set():
            try:
                # Wait for the specified interval before the next refresh
                time.sleep(self.sheet_refresh_interval)
                logger.info("Background thread: Refreshing sheets...")
                self._refresh_sheets(force=True)
                logger.info("Background thread: Sheets refreshed successfully.")
            except Exception as e:
                logger.error(f"Background sheet refresh failed: {e}")
                
    def _fetch_sheet_data(self, sheet_name: str) -> List[Dict[str, Any]]:
        params = {"action": "getSheetData", "sheet": sheet_name}
        print(f"[DEBUG] Fetching sheet {sheet_name}...")
        resp = self.http.get(self.sheet_api, params=params, timeout=self.sheet_fetch_timeout)
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Sheets error: {data['error']}")
        return data if isinstance(data, list) else []

    def _refresh_sheets(self, force=False):
        now = time.time()
        if not force and now - self.sheet_last_refresh < self.sheet_refresh_interval:
            return
        
        print("[DEBUG] Refreshing all sheet data...")
        self.qna_rows = self._fetch_sheet_data(self.qna_sheet) or []
        self.sheet_last_refresh = now
        print("[DEBUG] Sheet data refresh complete.")

    # ==================== CHAT HISTORY (Unchanged) ====================
    def add_chat_message(self, session_key: str, role: str, content: str, meta: dict = None):
        with self.chat_lock:
            if session_key not in self.chat_histories:
                self.chat_histories[session_key] = []
            self.chat_histories[session_key].append({
                "role": role,
                "content": content,
                "meta": meta or {},
                "timestamp": datetime.utcnow().isoformat() + "Z"
            })

    def get_recent_history(self, session_key: str) -> list:
        with self.chat_lock:
            return self.chat_histories.get(session_key, [])[-self.chat_history_limit:]

    def _format_conversation_for_prompt(self, history: list) -> str:
        if not history: return ""
        return "\n".join([f"{msg.get('role', '').title()}: {msg.get('content', '')}" for msg in history])

    # ==================== SESSION HANDLING (Unchanged) ====================
    def _extract_session_object(self, user_session, session_key):
        if not user_session:
            return None, None
        if isinstance(user_session, dict) and ("frontend" in user_session or "normalized" in user_session):
            norm = user_session.get("normalized", {}) or {}
            key = norm.get("email") or norm.get("client_id") or session_key
            return key, user_session
        if isinstance(user_session, dict) and session_key and session_key in user_session:
            return session_key, user_session[session_key]
        return None, None

    # ==================== PROMPT BUILDING (Unchanged) ====================
    # (Guest prompt commented out in original, leaving strictly as requested)

    def _build_non_guest_prompt(self, query: str, recent_conversation: str, agent_name: str, hotel_data, rules_text, campaigns_text, user_profile_block) -> str:
        # This function remains unchanged except for ensuring formatting.
        recent_conv_text = f"\n\nRecent Conversation:\n{recent_conversation}" if recent_conversation else ""
        prompt = (f"You are {agent_name}, a polite and helpful assistant at *ILORA RETREATS*.\n\n" f"Note: Give concise to the point answers which are helpful to the user query\n\n" f"Here is the user profile:\n{user_profile_block}\n\n" f"Recent Conversation:\n{recent_conv_text}\n\n"  f"ILORA RETREATS OVERVIEW:\n" f"Ilora Retreats is a luxury safari camp in Kenya's Masai Mara, near Olkiombo Airstrip. " f"We offer equipped luxury tents with en-suite bathrooms, private verandas, and modern amenities. " f"Our retreat features a pool, spa, gym, yoga facilities, and various safari activities including game drives, " f"walking safaris and Maasai cultural experiences.\n\n" f"NON-GUEST STATUS - LIMITED ACCESS:\n" f"This user is NOT currently a registered guest. You can help them with:\n" f"✓ General information about ILORA RETREATS\n" f"✓ Room types (14 luxury tents and 14 standard tents) and general availability\n" f"✓ Pricing ranges (USD 500-650/night full-board)\n" f"✓ Location and directions (Masai Mara, near Olkiombo Airstrip)\n" f"✓ Activities overview (safaris, cultural experiences)\n" f"✓ Facilities overview (spa, pool, gym, dining)\n" f"✓ Booking process and reservation assistance\n" f"✓ General inquiry handling\n\n" f"RESTRICTED - CANNOT ACCESS:\n" f"✗ Detailed menu prices or in-room dining options\n" f"✗ Cannot book specific spa treatments or room service\n" f"✗ Cannot make in-stay arrangements\n" f"✗ Cannot access guest-only services\n" f"✗ Cannot view or modify existing bookings\n\n" f"HOTEL DATA (Relevant Information):\n{hotel_data}\n\n" f"GUEST QUERY: {query}\n\n" f"IMPORTANT RULES:\n" f"✓ Be welcoming and encouraging about the hotel\n" f"✓ If they ask about guest-only services (room service, spa bookings etc), politely explain " f"they need to be a registered guest to access these services\n" f"✓ Encourage them to make a reservation for full access to amenities\n" f"Provide general pricing: Full-board rates start around USD 500–650 per night if asked\n" f"❌ Do NOT hallucinate. Stick to general facts about the retreat\n" f"✓ Be professional, friendly, and persuasive about the unique luxury safari experience\n" f"✓ Emphasize sustainability, comfort, and immersive nature experience\n\n" f"Provide a concise helpful response to the user")
        return prompt

    # ==================== LLM CALLS (UPDATED TO GEMINI) ====================
    def _call_llm_gemini(self, prompt: str, max_retries: int = 3) -> str:
        """
        Executes the prompt using Google Gemini 1.5 Pro.
        Handles model instantiation, generation config, and safety settings.
        """
        for attempt in range(max_retries):
            try:
                # Initialize model
                model = genai.GenerativeModel(self.llm_model_name)

                # Safety settings: Block few (optional - relax restrictions for hotel queries to avoid false positives)
                safety_settings = {
                    HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                    HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_ONLY_HIGH,
                }

                # Generation config
                generation_config = genai.types.GenerationConfig(
                    temperature=0.7,
                    max_output_tokens=1024, # Adjust if you want longer responses
                )

                # Generate content
                response = model.generate_content(
                    prompt,
                    generation_config=generation_config,
                    safety_settings=safety_settings
                )
                
                # Return text
                if response.text:
                    return response.text.strip()
                else:
                    logger.warning("Gemini returned empty response.")
                    return "I apologize, I couldn't generate a response."

            except Exception as e:
                logger.error(f"Gemini API failed on attempt {attempt + 1}: {e}")
                if "429" in str(e): # Rate limit
                    time.sleep(2 ** attempt)
                elif attempt == max_retries - 1:
                    return "I'm currently experiencing high traffic. Please ask me again in a moment."
        
        return "I'm having trouble processing that right now. Please try again."

    def _run_with_timeout(self, fn, args: tuple = (), timeout: float = 5.0):
        # This function remains unchanged.
        future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            future.cancel()
            raise TimeoutError(f"Operation timed out after {timeout}s")
            
    # ==================== MAIN ASK METHOD (OPTIMIZED) ====================
    def ask(self, query: str, user_type: str = "non-guest", user_session=None, session_key=None, user_data = None) -> str:
        print(f"[DEBUG] >>> ask: {query} (user_type={user_type})")

        sess_key, sess_obj = self._extract_session_object(user_session, session_key)

        # PERFORMANCE FIX: Background thread handles updates, so we proceed immediately.

        # Get agent name
        agent_name = "Welcome_AI_Agent" 
        try:
            agents_file = os.path.join("data", "agents.json")
            if os.path.exists(agents_file):
                with open(agents_file, "r", encoding="utf-8") as f:
                    agents = json.load(f)
                agent_name = next((a.get("agent_name", agent_name) for a in agents if a.get("Name") == "Front Desk"), agent_name)
        except Exception:
            pass
            
        # Prepare context from memory
        user_profile_text = user_data or ""
        recent_conversation = self._format_conversation_for_prompt(self.get_recent_history(sess_key)) if sess_key else ""
        
        rules_text = ""
        campaigns_text = ""

        # With Gemini's large context window, we can safely pass more data rows if needed.
        # Currently set to 500 rows as per original, but you could increase this to 2000+ with Gemini 1.5 Pro.
        hotel_data = "\n".join(str(row) for row in self.qna_rows[:500]) if self.qna_rows else "No specific data available."
        
        prompt = self._build_non_guest_prompt(query, recent_conversation, agent_name, hotel_data, rules_text, campaigns_text, user_profile_text)

        # Call LLM (Gemini)
        try:
            answer = self._call_llm_gemini(prompt)
        except Exception as e:
            logger.error(f"LLM call failed: {e}")
            answer = "I'm sorry, I couldn't process that right now. Please try again."

        # Save to chat history
        if sess_key:
            self.add_chat_message(sess_key, "user", query)
            self.add_chat_message(sess_key, "assistant", answer)

        return answer

    def shutdown(self):
        """Gracefully stops background threads and shuts down the executor."""
        logger.info("Shutting down the concierge bot...")
        self._stop_event.set()
        self._executor.shutdown(wait=True)
        logger.info("Shutdown complete.")