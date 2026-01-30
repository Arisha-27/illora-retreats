# qa_agent.py (FINAL: Smart Capacity Pricing Logic)
from typing import Optional, List, Dict, Any, Tuple
from langchain_openai import ChatOpenAI
from vector_store import create_vector_store
from config import Config
from logger import setup_logger
import os
import requests
import re
import threading
import csv
from io import StringIO
from datetime import datetime
from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeoutError

logger = setup_logger("QAAgent")
os.environ["TOKENIZERS_PARALLELISM"] = "false"

class ConciergeBot:
    def __init__(self):
        print("[DEBUG] Initializing ConciergeBot (Smart Pricing Mode)...")
        
        self.ADULT_PRICE = 1000
        self.CHILD_PRICE = 1000
        
        self.sheet_api = getattr(Config, "GSHEET_WEBAPP_URL", None)
        self.retriever_k = int(getattr(Config, "RETRIEVER_K", 5))
        self.retrieve_timeout = 15.0 
        self.llm_timeout = float(getattr(Config, "LLM_TIMEOUT", 15.0))
        self.http = requests.Session()

        self.llm = ChatOpenAI(
            api_key=Config.OPENAI_API_KEY,
            model=Config.OPENAI_MODEL,
            base_url=Config.GROQ_API_BASE,
            temperature=0, 
        )

        self._executor = ThreadPoolExecutor(max_workers=4)
        self.chat_histories: Dict[str, List[Dict[str, Any]]] = {}
        self.chat_lock = threading.Lock()
        self.chat_history_limit = 10

        try:
            self.vector_store = create_vector_store()
            self.retriever = self.vector_store.as_retriever(search_kwargs={"k": 5})
            logger.info("Local FAISS retriever loaded.")
        except Exception as e:
            logger.error(f"Vector store init failed: {e}")
            self.retriever = None

        logger.info("ILORA RETREATS ConciergeBot ready.")

    # --- HELPER: Identify Season ---
    def _get_current_season_name(self):
        now = datetime.now()
        m, d = now.month, now.day
        # Peak: Jul-Oct OR Dec 20 - Jan 5
        if (7 <= m <= 10) or (m == 12 and d >= 20) or (m == 1 and d <= 5):
            return "Peak Season"
        # Off: Apr-May
        if (4 <= m <= 5):
            return "Off Season"
        return "Regular"

    # --- HELPER: Smart Pricing Logic ---
    def _get_pricing_context(self, query):
        if not self.sheet_api: return ""

        try:
            # 1. Fetch CSV
            resp = self.http.get(self.sheet_api, timeout=10)
            if resp.status_code != 200: return ""
            
            current_season = self._get_current_season_name()
            reader = csv.DictReader(StringIO(resp.text))
            
            # 2. Detect Guests
            adults_match = re.search(r"(\d+)\s*(?:adult|guest|person|pax|people)", query.lower())
            child_match = re.search(r"(\d+)\s*(?:child|kid|infant)", query.lower())
            
            num_adults = int(adults_match.group(1)) if adults_match else 0
            num_children = int(child_match.group(1)) if child_match else 0
            total_guests = num_adults + num_children

            # If no count, we can't do the "Smart Logic", so return generic info
            if total_guests == 0:
                # Basic check if user asked for price
                if any(x in query.lower() for x in ["price", "cost", "how much", "rate"]):
                    return f"\n📊 **Standard Rates ({current_season}):**\nPlease specify the number of adults and children for an exact quote."
                return ""

            lines = [f"\n💰 **QUOTE FOR {num_adults} ADULTS, {num_children} CHILDREN ({current_season}):**"]
            
            valid_rows = 0
            
            for row in reader:
                # Clean keys
                clean_row = {re.sub(r"[^a-z0-9]", "", str(k).lower()): v for k, v in row.items()}
                
                rtype = row.get("Room Type", clean_row.get("roomtype", "Room"))
                rseason = row.get("Season", clean_row.get("season", ""))
                
                # Filter by Season
                if current_season.lower() not in rseason.lower() and rseason.lower() not in current_season.lower():
                    continue

                # Get Prices (Handle various column names)
                try:
                    # Final Price (The "Deal" Price)
                    final_str = clean_row.get("finalprice", "") or clean_row.get("finalpriceinr", "")
                    final_price = float(re.sub(r"[^\d.]", "", final_str)) if final_str else 0
                    
                    # Base Price (The "Rack" Price for calculation)
                    base_str = clean_row.get("baseprice", "") or clean_row.get("basepriceinr", "")
                    base_price = float(re.sub(r"[^\d.]", "", base_str)) if base_str else final_price
                except:
                    continue

                # --- THE LOGIC CORE ---
                
                # 1. Determine Capacity
                # If "Family", "Suite", "Presidential" -> Capacity 4. Else -> Capacity 2.
                capacity = 4 if any(x in rtype.lower() for x in ["family", "suite", "presidential"]) else 2
                
                total_cost = 0
                calculation_note = ""

                # 2. Apply The Rules
                if total_guests <= capacity:
                    # Scenario A: Within Capacity -> Return Final Price (Flat Rate)
                    total_cost = final_price
                    calculation_note = f"(Standard Rate for up to {capacity} pax)"
                else:
                    # Scenario B: Exceeds Capacity -> Base + (1000 * Adults) + (1000 * Kids)
                    guest_charge = (num_adults * self.ADULT_PRICE) + (num_children * self.CHILD_PRICE)
                    total_cost = base_price + guest_charge
                    calculation_note = f"(Base ₹{int(base_price)} + Guest Charges ₹{guest_charge})"

                lines.append(f"- **{rtype}**: ₹{int(total_cost)}")
                lines.append(f"  _{calculation_note}_")
                valid_rows += 1

            if valid_rows == 0:
                return ""
            
            return "\n".join(lines)

        except Exception as e:
            print(f"[WARN] Pricing Engine Error: {e}")
            return ""

    # --- STANDARD HELPERS ---
    def _extract_session_object(self, user_session, session_key):
        if not user_session: return None, None
        if isinstance(user_session, dict):
            norm = user_session.get("normalized", {}) or {}
            key = norm.get("email") or norm.get("client_id") or session_key
            return key, user_session
        return None, None

    def get_recent_history(self, session_key: str) -> list:
        with self.chat_lock:
            return self.chat_histories.get(session_key, [])[-self.chat_history_limit:]

    def add_chat_message(self, session_key: str, role: str, content: str, meta: dict = None):
        with self.chat_lock:
            if session_key not in self.chat_histories: self.chat_histories[session_key] = []
            self.chat_histories[session_key].append({"role": role, "content": content, "meta": meta or {}})

    def _format_conversation_for_prompt(self, history: list) -> str:
        if not history: return ""
        lines = [f"{msg.get('role', '').title()}: {msg.get('content', '')}" for msg in history[-10:]]
        return "\n".join(lines)

    def _run_with_timeout(self, fn, args: tuple = (), timeout: float = 5.0):
        future = self._executor.submit(fn, *args)
        try:
            return future.result(timeout=timeout)
        except FuturesTimeoutError:
            try: future.cancel()
            except: pass
            raise TimeoutError

    # ---------------- MAIN ASK FUNCTION ----------------
    def ask(self, query: str, user_type=None, user_session=None, session_key=None, user_data = None) -> str:
        print(f"[DEBUG] >>> ask: {query}")
        sess_key, sess_obj = self._extract_session_object(user_session, session_key)

        # 1. Retrieve Q&A
        docs = []
        if getattr(self, "retriever", None):
            try:
                docs_retrieved = self._run_with_timeout(lambda q: self.retriever.invoke(q), (query,), timeout=self.retrieve_timeout)
                docs = [{"page_content": doc.page_content} for doc in docs_retrieved]
            except: pass

        # 2. CALCULATE PRICING
        rate_context = self._get_pricing_context(query)

        # 3. Build Prompt
        hotel_data = "\n".join(d["page_content"] for d in docs[:4]) if docs else "No specific Q&A found."
        recent_conv = self._format_conversation_for_prompt(self.get_recent_history(sess_key)) if sess_key else ""
        
        prompt = (
            f"You are the AI Concierge at Ilora Retreats.\n"
            f"--- CALCULATED QUOTE ---\n{rate_context}\n"
            f"--- GENERAL INFO ---\n{hotel_data}\n"
            f"------------------------\n\n"
            f"INSTRUCTIONS:\n"
            f"1. Use the 'CALCULATED QUOTE' numbers exactly. They are pre-calculated based on the guest count.\n"
            f"2. Present the options clearly so the user can choose (e.g., 'For your group of X, here are your options...').\n"
            f"3. If no guest count was provided in the query, ask for it politely to give an accurate rate.\n"
            f"4. Quote in Rupees (₹).\n\n"
            f"Chat History:\n{recent_conv}\n\n"
            f"Guest Query: {query}\n"
            f"Response:"
        )

        try:
            resp = self._run_with_timeout(lambda: self.llm.invoke(prompt), timeout=self.llm_timeout)
            answer = resp.content if hasattr(resp, "content") else str(resp)
        except Exception as e:
            print(f"[ERROR] LLM Error: {e}")
            answer = "I'm having trouble calculating that right now. Please ask our Front Desk."

        if sess_key:
            self.add_chat_message(sess_key, "user", query, meta={"ts": datetime.utcnow().isoformat() + "Z"})
            self.add_chat_message(sess_key, "assistant", answer, meta={"ts": datetime.utcnow().isoformat() + "Z"})

        return answer
    
    def shutdown(self):
        self._executor.shutdown(wait=True)