## importing essential libraries:

from flask import Flask, request
from twilio.twiml.messaging_response import MessagingResponse
from Hotel_AI_Bot import IloraRetreatsConciergeBot
from services.payment_gateway import create_checkout_session, create_addon_checkout_session
from services.google_sheets_service import GoogleSheetsService
from logger import log_chat, setup_logger
from services.intent_classifier import classify_intent
from config import Config
import uuid
import json
import os
import hashlib
import re
from datetime import datetime, timedelta
import os
import uuid
import json
import random
import logging
from typing import List, Optional, Dict, Any
from datetime import datetime
import httpx
import requests
import re
from fastapi import FastAPI, HTTPException, Body, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from config import Config
# Project imports (kept)
import web_ui_final as web
from services.intent_classifier import classify_intent
from logger import log_chat
from services.qa_agent import ConciergeBot
from services.payment_gateway import (
    create_checkout_session,
    create_addon_checkout_session,
    create_pending_checkout_session,
)
# Illora checkin app / models
from illora.checkin_app.models import Room, Booking, BookingStatus
from illora.checkin_app.pricing import calculate_price_for_room as calculate_price
from illora.checkin_app.database import Base, engine, SessionLocal
from illora.checkin_app.booking_flow import create_booking_record
from illora.checkin_app.chat_models import ChatMessage
from sqlalchemy import func
from sqlalchemy.orm import Session
from datetime import datetime
from typing import Tuple, Optional, Dict, Any
import os
import uuid
import json
import random
import logging
import asyncio
import requests
from typing import List, Optional, Dict, Any, Generator
from datetime import date, datetime, timedelta

from fastapi import FastAPI, Depends, UploadFile, File, HTTPException, Query, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field
from config import Config

room_alloted = ''
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

## --pyndatic models 
class ChatReq(BaseModel):
    message: str
    is_guest: Optional[bool] = False
    session_id: Optional[str] = None
    email: Optional[str] = None

class ChatActions(BaseModel):
    show_booking_form: bool = False
    addons: List[str] = Field(default_factory=list)
    payment_link: Optional[str] = None
    pending_balance: Optional[Dict[str, Any]] = None

class ChatResp(BaseModel):
    reply: str
    reply_parts: Optional[List[str]] = None
    intent: Optional[str] = None
    actions: ChatActions = Field(default_factory=ChatActions)
    media_url: Optional[str] = None 
# ----------

# ------------------------- FastAPI app -------------------------
app = FastAPI(title="Ilora Retreats API", version="2.0.0")

CLIENT_WORKFLOW_SHEET = "Client_workflow"

# ------------------------- CORS -------------------------
FRONTEND_ORIGINS = [
    "http://localhost:8080",
    "http://localhost:3000",
    "http://localhost:5173",
    "http://127.0.0.1:5173",
    "https://ai-chieftain.webisdomtech.com"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=FRONTEND_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
    allow_headers=["*"],
    expose_headers=["*"],
)

# Load room prices from configuration
try:
    with open(os.path.join("data", "room_config.json"), "r") as f:
        config = json.load(f)
        ROOM_PRICES = config.get("room_prices", {
            "Luxury Tent": 50000  # Base price per night in INR
        })
        TOTAL_TENTS = config.get("total_tents", 14)
except Exception as e:
    logger.warning(f"Could not load room config, using defaults: {e}")
    ROOM_PRICES = {"Luxury Tent": 50000}
    TOTAL_TENTS = 14

ROOM_OPTIONS = list(ROOM_PRICES.keys())

ADDON_MAPPING = {
    "spa": "spa",
    "massage": "spa",
    "hot air balloon": "hot_air_balloon",
    "balloon ride": "hot_air_balloon",
    "game drive": "game_drive",
    "safari": "game_drive",
    "walking safari": "walking_safari",
    "bush dinner": "bush_dinner",
    "maasai cultural": "maasai_experience",
    "stargazing": "stargazing"
}

#  Guest-only services
GUEST_ONLY_SERVICES = [
    "room service", "in-room", "spa", "swimming pool", "pool access",
    "gym", "yoga", "bush dinner", "stargazing", "game drive", "safari"
]

###################  helper functions ########################################
def hash_password(password):
    """Hash password using SHA256"""
    return hashlib.sha256(password.encode()).hexdigest()

def validate_email(email):
    """Validate email format"""
    pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
    return re.match(pattern, email) is not None

def validate_date(date_str):
    """Validate date format DD-MM-YYYY"""
    try:
        datetime.strptime(date_str, "%d-%m-%Y")
        return True
    except ValueError:
        return False

def send_media_message(msg, media_url, caption=""):
    """Helper function to send media with caption"""
    actions = ChatActions()
    intent = 'media_see'
    bot_reply_text = f"Here is our {caption}"
    # media_url = "https://example.com/image.jpg"
    # log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
    reply_parts = str(bot_reply_text).split("\n\n") if isinstance(bot_reply_text, str) else [str(bot_reply_text)]
    return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions, media_url=media_url) 

def get_latest_session(user_sessions: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Return (session_key, session_obj) for the session with the newest last_login.
    If none found, returns (None, None).
    """
    if not user_sessions:
        return None, None

    latest_key = None
    latest_ts = None

    for key, sess in user_sessions.items():
        last_login_str = sess.get("last_login")
        if last_login_str:
            try:
                # handle trailing Z
                ts = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
            except Exception:
                ts = None
        else:
            ts = None

        # treat missing timestamps as very old so they won't override real ones
        if ts is None:
            continue

        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_key = key

    # If we didn't find any with parseable timestamps but there are sessions, pick an arbitrary one
    if latest_key is None and len(user_sessions) > 0:
        # pick last inserted key (stable for dicts in Python 3.7+)
        try:
            latest_key = next(reversed(user_sessions))
        except Exception:
            latest_key = next(iter(user_sessions))

    if latest_key:
        return latest_key, user_sessions.get(latest_key)
    return None, None


###########################################################################
# ------------------------- Helpers / debug utils -------------------------
def _normalize_key(k: Any) -> str:
    return "".join(ch.lower() for ch in str(k) if ch.isalnum())

def _parse_float(val: Any) -> float:
    if val is None or str(val).strip() == "":
        return 0.0
    try:
        s = str(val).strip().replace("$", "").replace(",", "")
        return float(s)
    except Exception:
        return 0.0

def _short(s: str, n: int = 400) -> str:
    if s is None:
        return ""
    s = str(s)
    return s if len(s) <= n else s[:n] + " ...[truncated]"

def get_first_value(d: Dict[str, Any], candidates: List[str], default: Any = "") -> Any:
    if not d:
        return default
    for k in candidates:
        if k in d and d[k] not in (None, ""):
            return d[k]
    lowered = {str(k).lower(): v for k, v in d.items() if v not in (None, "")}
    for k in candidates:
        if k.lower() in lowered:
            return lowered[k.lower()]
    return default

def map_sheet_row_to_user_details(row: Dict[str, Any]) -> Dict[str, Any]:
    """Map exact columns to the frontend-friendly object used by HotelSidebar."""
    if not row:
        return {}
    # Keep exact keys as in sheet when useful
    uid = row.get("Client Id", "") or row.get("ClientId", "") or row.get("client_id", "")
    booking_status = row.get("Workfow Stage", "") or row.get("Workflow Stage", "") or row.get("Booking Status", "") or "Not Booked"
    # id proof may be a status or a link
    id_proof = row.get("Id Link", "") or row.get("IdLink", "") or row.get("ID Proof", "") or ""
    pending_balance = _parse_float(row.get("Pending Balance", 0) or row.get("Balance", 0) or 0)
    status = row.get("Status", "") or booking_status or "Still"
    room_number = row.get("Room Alloted", "") or row.get("Room Number", "") or ""
    check_in = row.get("CheckIn", "") or row.get("Check In", "")
    check_out = row.get("Check Out", "") or row.get("CheckOut", "")

    return {
        "uid": uid,
        "bookingStatus": booking_status,
        "bookingId": row.get("Booking Id", "") or row.get("BookingId", "") or "",
        "idProof": id_proof,
        "pendingBalance": pending_balance,
        "status": status,
        "roomNumber": room_number,
        "checkIn": check_in,
        "checkOut": check_out,
        "_raw_row": {k: (v if _normalize_key(k) != "password" else "****") for k, v in (row or {}).items()}
    }

def normalize_raw_user_data(raw_user_data: Dict[str, Any]) -> Dict[str, Any]:
    """Normalized snake_case mapping (useful for other endpoints)."""
    return {
        "client_id": raw_user_data.get("Client Id", "") or raw_user_data.get("ClientId", ""),
        "name": raw_user_data.get("Name", "") or raw_user_data.get("Full Name", ""),
        "email": raw_user_data.get("Email", ""),
        "booking_id": raw_user_data.get("Booking Id", ""),
        "workflow_stage": raw_user_data.get("Workfow Stage", "") or raw_user_data.get("Workflow Stage", ""),
        "room_alloted": raw_user_data.get("Room Alloted", ""),
        "check_in": raw_user_data.get("CheckIn", "") or raw_user_data.get("Check In", ""),
        "check_out": raw_user_data.get("Check Out", "") or raw_user_data.get("CheckOut", ""),
        "id_link": raw_user_data.get("Id Link", ""),
        "pending_balance": _parse_float(raw_user_data.get("Pending Balance", 0)),
        "status": raw_user_data.get("Status", "") or raw_user_data.get("Workfow Stage", "")
    }

def _update_session_from_raw(username: str, raw_user_data: Dict[str, Any], remember_token: Optional[str] = None):
    normalized = normalize_raw_user_data(raw_user_data)
    frontend_view = map_sheet_row_to_user_details(raw_user_data)
    USER_SESSIONS[username] = {
        "normalized": normalized,
        "raw": raw_user_data,
        "frontend": frontend_view,
        "last_login": datetime.utcnow().isoformat() + "Z",
        "remember_token": remember_token or USER_SESSIONS.get(username, {}).get("remember_token"),
    }
    logger.debug("Session for %s updated/saved (sanitized): %s", username, json.dumps({
        "normalized": normalized,
        "frontend": frontend_view,
        "last_login": USER_SESSIONS[username]["last_login"],
        "remember_token": USER_SESSIONS[username]["remember_token"]
    }, default=str))

def get_latest_session(user_sessions: Dict[str, Any]) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
    """
    Return (session_key, session_obj) for the session with the newest last_login.
    If none found, returns (None, None).
    """
    if not user_sessions:
        return None, None

    latest_key = None
    latest_ts = None

    for key, sess in user_sessions.items():
        last_login_str = sess.get("last_login")
        if last_login_str:
            try:
                # handle trailing Z
                ts = datetime.fromisoformat(last_login_str.replace("Z", "+00:00"))
            except Exception:
                ts = None
        else:
            ts = None

        # treat missing timestamps as very old so they won't override real ones
        if ts is None:
            continue

        if latest_ts is None or ts > latest_ts:
            latest_ts = ts
            latest_key = key

    # If we didn't find any with parseable timestamps but there are sessions, pick an arbitrary one
    if latest_key is None and len(user_sessions) > 0:
        # pick last inserted key (stable for dicts in Python 3.7+)
        try:
            latest_key = next(reversed(user_sessions))
        except Exception:
            latest_key = next(iter(user_sessions))

    if latest_key:
        return latest_key, user_sessions.get(latest_key)
    return None, None

###########################################################################
# ------------------------- Sheets helpers (with debug) -------------------------
def push_row_to_sheet(sheet_name: str, row_data: Dict[str, Any]) -> Dict[str, Any]:
    if not Config.GSHEET_WEBAPP_URL:
        raise RuntimeError("GSHEET_WEBAPP_URL not configured in Config")

    safe_preview = {k: (v if _normalize_key(k) != "password" else "****") for k, v in row_data.items() if k in ("Client Id", "Email", "Name", "Password")}
    logger.debug("push_row_to_sheet: payload preview: %s", safe_preview)
    payload = {"action": "addRow", "sheet": sheet_name, "rowData": row_data}
    try:
        resp = requests.post(Config.GSHEET_WEBAPP_URL, json=payload, timeout=15, allow_redirects=True)
        logger.debug("push_row_to_sheet: response url=%s status=%s", getattr(resp, "url", None), resp.status_code)
        text = _short(resp.text, 1000)
        logger.debug("push_row_to_sheet: response text (truncated) = %s", text)
        resp.raise_for_status()
        try:
            return resp.json()
        except ValueError:
            return {"success": resp.ok, "status_code": resp.status_code, "text": text}
    except Exception as e:
        logger.exception("push_row_to_sheet: exception while calling sheet")
        return {"success": False, "message": str(e)}

def fetch_client_row_from_sheet_by_email(email: str) -> Optional[Dict[str, Any]]:
    if not Config.GSHEET_WEBAPP_URL:
        logger.error("GSHEET_WEBAPP_URL not configured")
        return None

    try:
        params = {"action": "getSheetData", "sheet": CLIENT_WORKFLOW_SHEET}
        logger.debug("fetch_client_row_from_sheet_by_email: calling GET %s params=%s", Config.GSHEET_WEBAPP_URL, params)
        resp = requests.get(Config.GSHEET_WEBAPP_URL, params=params, timeout=15, allow_redirects=True)
        logger.debug("fetch_client_row_from_sheet_by_email: response status=%s url=%s", resp.status_code, getattr(resp, "url", None))
        try:
            rows = resp.json()
        except ValueError:
            logger.error("fetch_client_row_from_sheet_by_email: getSheetData returned non-JSON: %s", _short(resp.text, 800))
            return None

        if not isinstance(rows, list):
            logger.error("fetch_client_row_from_sheet_by_email: unexpected sheet response type=%s", type(rows))
            return None

        target = (email or "").strip().lower()
        for idx, row in enumerate(rows):
            for key, val in (row or {}).items():
                if _normalize_key(key) in ("email", "username"):
                    if str(val or "").strip().lower() == target:
                        logger.debug("fetch_client_row_from_sheet_by_email: found matching row index=%s, row=%s", idx, {k: (v if _normalize_key(k) != "password" else "****") for k, v in row.items()})
                        return row
        logger.debug("fetch_client_row_from_sheet_by_email: no match for %s", target)
        return None
    except Exception as e:
        logger.exception("Error fetching sheet for email lookup")
        return None

# ------------- Generic helper to push a row to any sheet via your Apps Script Web App -------------
def push_row_to_sheet(sheet_name: str, row_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Call Apps Script webapp with: { action: 'addRow', sheet: sheet_name, rowData: {...} }.
    Returns parsed JSON from webapp or raises on error.
    """
    if not GSHEET_WEBAPP_URL:
        raise RuntimeError("GSHEET_WEBAPP_URL not configured in Config")

    payload = {"action": "addRow", "sheet": sheet_name, "rowData": row_data}
    resp = requests.post(GSHEET_WEBAPP_URL, json=payload, timeout=10)
    resp.raise_for_status()
    try:
        return resp.json()
    except Exception:
        return {"status_code": resp.status_code, "text": resp.text}

# ------------- Guest interaction logging helper -------------
def _naive_sentiment(message: str) -> str:
    """Very small sentiment heuristic (optional). Returns 'positive' / 'negative' / ''."""
    if not message:
        return ""
    m = message.lower()
    negative_words = ["not", "no", "never", "bad", "disappointed", "angry", "hate", "worst", "problem", "issue", "delay"]
    positive_words = ["good", "great", "awesome", "excellent", "happy", "love", "enjoy"]
    if any(w in m for w in negative_words) and not any(w in m for w in positive_words):
        return "negative"
    if any(w in m for w in positive_words) and not any(w in m for w in negative_words):
        return "positive"
    return ""

def create_guest_log_row(req_session_id: Optional[str], email: Optional[str], user_input: str, bot_response: str,
                         intent: str, is_guest_flag: bool, ref_ticket_id: Optional[str] = None) -> Dict[str, Any]:
    """
    Build a row matching headers:
    Log ID | Timestamp | Source | Session ID | Guest Email | Guest Name | User Input | Bot Response | Intent | Guest Type | Sentiment | Reference Ticket ID | Conversation URL
    """
    log_id = f"LOG-{random.randint(1000,999999)}"
    timestamp = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    source = "web"
    session_id = req_session_id or ""
    guest_email = email or ""
    guest_name = "Guest"
    user_input_val = user_input or ""
    bot_response_val = bot_response or ""
    intent_val = intent or ""
    guest_type = "guest" if bool(is_guest_flag) else "non-guest"
    sentiment = _naive_sentiment(user_input)
    reference_ticket_id = ref_ticket_id or ""
    conversation_url = ""  # optional — left blank or build if you have a UI link

    return {
        "Log ID": log_id,
        "Timestamp": timestamp,
        "Source": source,
        "Session ID": session_id,
        "Guest Email": guest_email,
        "Guest Name": guest_name,
        "User Input": user_input_val,
        "Bot Response": bot_response_val,
        "Intent": intent_val,
        "Guest Type": guest_type,
        "Sentiment": sentiment,
        "Reference Ticket ID": reference_ticket_id,
        "Conversation URL": conversation_url,
    }

def is_ticket_request(message: str, intent: str, addon_matches: list = None) -> bool:
    """
    Return True when message likely requests a service/add-on that should create a ticket.
    Combines intent signals + simple keyword scanning + menu addon hints.
    """
    if not message:
        return False
    lower = message.lower()

    # Strong intents from your classifier that clearly map to service requests
    ticket_intents = {
        "book_addon_spa",
        "book_addon_beverage",
        "book_addon_food",
        "request_service",
        "room_service_request",
        "maintenance_request",
        "order_addon",
        "wake-up-call",
        "urgent_assistance",
        # (add other classifier intent names you use)
    }
    if intent in ticket_intents:
        return True

    # Keyword-based fallback (covers "order a coffee", "please bring towel", "fix ac", etc.)
    keywords = [
        "coffee", "tea", "order", "bring", "deliver", "room service", "food", "meal", "snack",
        "towel", "clean", "housekeeping", "makeup room", "turn down", "repair", "fix", "ac", "wifi",
        "tv", "light", "broken", "leak", "toilet", "bathroom", "shower", "wake-up-call","pickup and drop","laundry","taxi","transportation", "request", "need", "help", "assist", "urgent"
    ]
    if any(k in lower for k in keywords):
        return True

    # If we already matched menu addons (e.g., "club sandwich" matched), that's also a ticket trigger
    if addon_matches and len(addon_matches) > 0:
        return True

    return False

def classify_ticket_category(message: str) -> str:
    """Map message content to a ticket category."""
    m = message.lower()
    if any(w in m for w in ["coffee", "tea", "drink", "food", "meal", "snack", "beverage", "breakfast", "lunch", "dinner"]):
        return "Food"
    if any(w in m for w in ["towel", "clean", "housekeeping", "room service", "bed", "makeup", "turn down", "linen"]):
        return "Room Service"
    if any(w in m for w in ["ac", "wifi", "tv", "light", "repair", "engineer", "fix", "leak", "broken", "toilet", "plumb", "electr"]):
        return "Engineering"
    return "General"

def assign_staff_for_category(category: str) -> str:
    return {
        "Food": "Food Staff",
        "Room Service": "Room Service",
        "Engineering": "Engineering",
        "General": "Front Desk"
    }.get(category, "Front Desk")

def create_ticket_row_payload(message: str, email: str = None) -> Dict[str, str]:
    """
    Build the exact rowData dict matching your sheet's headers:
    Ticket ID | Guest Name | Room No | Request/Query | Category | Assigned To | Status | Created At | Resolved At | Notes
    
    Note: This function now tries to get the actual room number from Client_workflow sheet if possible
    """

    # get the latest session (key + object)
    session_key, session_obj = get_latest_session(USER_SESSIONS)
    # Get actual room number from Client_workflow sheet if possible

    print("\nSelected session_key:", session_key)
    print("Selected session_obj (sanitized):")
    if session_obj:
        # mask password in raw if present for logging
        raw_preview = {k: ("****" if re.sub(r"[^a-zA-Z0-9]", "", str(k)).lower() == "password" else v)
                       for k, v in (session_obj.get("raw", {}) or {}).items()}
        print("normalized:", session_obj.get("normalized"))
        print("frontend:", session_obj.get("frontend"))
        print("raw (masked):", raw_preview)
    else:
        print("No session selected (session_obj is None)")
    print()



    normalized = session_obj.get("normalized") if session_obj else {}

    guest_email = normalized.get("email") if isinstance(normalized, dict) else None
    room_no = normalized.get("room_alloted") if isinstance(normalized, dict) else None

    
    ticket_id = f"TCK-{random.randint(1000, 99999)}"
    guest_name = guest_email 
    category = classify_ticket_category(message)
    assigned_to = assign_staff_for_category(category)
    status = "In Progress"
    created_at = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S")
    resolved_at = ""  # empty initially
    notes = message

    # Use exact header names present in the spreadsheet (case + spaces matter for the Apps Script mapping)
    # Return the ticket data with the room number
    return {
        "Ticket ID": ticket_id,
        "Guest Name": guest_email,
        "Room No": room_no,  # Now using actual room number from Client_workflow sheet
        "Request/Query": message,
        "Category": category,
        "Assigned To": assigned_to,
        "Status": status,
        "Created At": created_at,
        "Resolved At": resolved_at,
        "Notes": notes
    }

def push_ticket_to_sheet(row_data: Dict[str, str]) -> Dict:
    """
    Call your Apps Script webapp with { action: 'addRow', sheet: <sheetName>, rowData: {...} }.
    Returns the parsed JSON response or raises on network error.
    """
    if not GSHEET_WEBAPP_URL:
        raise RuntimeError("GSHEET_WEBAPP_URL not configured in Config")

    payload = {
        "action": "addRow",
        "sheet": TICKET_SHEET_NAME,
        "rowData": row_data
    }
    try:
        resp = requests.post(GSHEET_WEBAPP_URL, json=payload, timeout=10)
        # Apps Script returns JSON like { success: true, message: 'Row added successfully' }
        try:
            return resp.json()
        except Exception:
            resp.raise_for_status()
            return {"ok": True, "status_code": resp.status_code}
    except Exception as e:
        # bubble up error to caller for graceful logging
        raise

def _fetch_sheet_data(self, sheet_name: str) -> List[Dict[str, Any]]:
    """
    Calls the deployed Apps Script web app and returns the list of row objects for the sheet.
    The web app must implement ?action=getSheetData&sheet=<sheetName> (matching your provided Apps Script).
    """
    if not self.sheet_api:
        raise RuntimeError("GSHEET_WEBAPP_URL is not configured in Config.")

    params = {"action": "getSheetData", "sheet": sheet_name}
    try:
        resp = requests.get(self.sheet_api, params=params, timeout=15)
        resp.raise_for_status()
        data = resp.json()
        # Data should be a list of objects (one per row). If the webapp returns {error:...}, raise.
        if isinstance(data, dict) and data.get("error"):
            raise RuntimeError(f"Sheets webapp returned error: {data.get('error')}")
        if not isinstance(data, list):
            # sometimes the webapp might wrap results; be permissive
            raise RuntimeError("Unexpected sheet response format (expected list of row objects).")
        return data
    except Exception as e:
        logger.error(f"Error fetching sheet '{sheet_name}' from {self.sheet_api}: {e}")
        raise

###########################################################################


# ------------------------- Static files -------------------------
STATIC_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static")
os.makedirs(STATIC_DIR, exist_ok=True)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# ------------------------- Concierge bot -------------------------
inroom_bot = ConciergeBot()
outroom_bot = IloraRetreatsConciergeBot()
session_data = {}
sheets_service = GoogleSheetsService()

# In-memory user session store
USER_SESSIONS: Dict[str, Dict[str, Any]] = {}

### models for upload workflow and requests:
class UpdateWorkflowReq(BaseModel):
    username: str
    stage: str
    booking_id: Optional[str] = None
    id_proof_link: Optional[str] = None

class MeReq(BaseModel):
    username: Optional[str] = None
    remember_token: Optional[str] = None

DEMO_ROOM_TYPES = ["Luxury Tent"]
sample_bookings: List[Dict[str, Any]] = []

TICKET_SHEET_NAME = getattr(Config, "GSHEET_TICKET_SHEET", "ticket_management")
GSHEET_WEBAPP_URL = getattr(Config, "GSHEET_WEBAPP_URL", "https://script.google.com/macros/s/AKfycbwfh2HvU5E0Y0Ruv5Ylfwdh524c0PWLCU0NduferN4etm08ovIMO6WoFoJVszmQx__O/exec")
GUEST_LOG_SHEET_NAME = getattr(Config, "GSHEET_GUEST_LOG_SHEET", "guest_interaction_log")
MENU_SHEET_NAME = getattr(Config, "GSHEET_MENU_SHEET", "menu_manager")
menu_rows: List[Dict[str, Any]] = []

# ------------------------- SSE Broker -------------------------
class EventBroker:
    def __init__(self):
        self.connections: List[asyncio.Queue] = []

    async def connect(self) -> asyncio.Queue:
        q: asyncio.Queue = asyncio.Queue()
        self.connections.append(q)
        return q

    async def disconnect(self, q: asyncio.Queue):
        if q in self.connections:
            try:
                self.connections.remove(q)
            except Exception:
                pass

    async def broadcast(self, event: str, data: Dict[str, Any]):
        msg = json.dumps({"event": event, "data": data}, default=str)
        for q in list(self.connections):
            try:
                await q.put(msg)
            except Exception:
                try:
                    self.connections.remove(q)
                except Exception:
                    pass

broker = EventBroker()

@app.get("/events")
async def sse_events(request: Request):
    async def event_generator(q: asyncio.Queue):
        try:
            await q.put(json.dumps({"event": "connected", "data": {}}))
            while True:
                if await request.is_disconnected():
                    break
                msg = await q.get()
                yield f"data: {msg}\n\n"
        finally:
            await broker.disconnect(q)

    q = await broker.connect()
    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_generator(q), headers=headers, media_type="text/event-stream")



# ------------------------- Pydantic Models -------------------------
class ChatReq(BaseModel):
    message: str
    is_guest: Optional[bool] = False
    session_id: Optional[str] = None
    email: Optional[str] = None

class ChatActions(BaseModel):
    show_booking_form: bool = False
    addons: List[str] = Field(default_factory=list)
    payment_link: Optional[str] = None
    pending_balance: Optional[Dict[str, Any]] = None

class ChatResp(BaseModel):
    reply: str
    reply_parts: Optional[List[str]] = None
    intent: Optional[str] = None
    actions: ChatActions = Field(default_factory=ChatActions)
    media_urls: Optional[List[str]] = None

menu = []
room_alloted = ''

@app.post("/chat", response_model=ChatResp)
async def chat(req: ChatReq):
    try:
        # Initialize variables
        pending_balance = 0.0
        user_input = req.message or ""
        print(f"[REQUEST] Message: {user_input}, Email: {req.email}, Session: {req.session_id}")
        
        # CRITICAL FIX: Use email-based session key instead of get_latest_session
        # This ensures each user has their own isolated session
        if req.email:
            session_key = req.email.lower()  # Use email as primary session key
        elif req.session_id:
            session_key = req.session_id
        else:
            session_key = f"anon_{datetime.now().timestamp()}"
        
        print(f"[SESSION] Using session_key: {session_key}")
        
        # Get or create session object for this specific user
        if session_key not in USER_SESSIONS:
            USER_SESSIONS[session_key] = {
                "normalized": {},
                "frontend": {},
                "raw": {},
                "created_at": datetime.now().isoformat()
            }
        
        session_obj = USER_SESSIONS[session_key]
        
        # Classify intent and initialize actions
        intent = classify_intent(user_input)
        actions = ChatActions()
        
        # Initialize session data for this specific user
        if session_key not in session_data:
            session_data[session_key] = {
                "stage": "welcome",
                "attempts": 0,
                "session_key": session_key  # Store for reference
            }
        
        # Get user-specific session
        user_session = session_data[session_key]
        stage = user_session.get("stage", "welcome")
        incoming_msg = user_input
        user_number = session_key
        
        print(f"[SESSION STATE] Stage: {stage}, User: {session_key}")
        
        # ==================== AUTHENTICATION FLOW ====================
        
        # Stage 0: Welcome
        if stage == "welcome":
            response = (
                "🌿 *Welcome to ILORA RETREATS* 🌿\n\n"
                "Your gateway to luxury safari experiences in Kenya's Masai Mara.\n\n"
                "To get started, please provide your *email address* to continue."
            )
            user_session["stage"] = "email_input"
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
            reply_parts = response.split("\n\n") if isinstance(response, str) else [str(response)]
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

        # Stage 1: Email Input
        elif stage == "email_input":
            if not validate_email(incoming_msg):
                response = "❌ Invalid email format. Please provide a valid email address (e.g., user@example.com)."
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
            
            email_lower = incoming_msg.lower()
            user_session["email"] = email_lower
            
            # CRITICAL: Update session_key to email if it was anonymous
            if session_key != email_lower:
                # Move session data to email-based key
                session_data[email_lower] = user_session
                USER_SESSIONS[email_lower] = session_obj
                # Clean up old keys
                if session_key in session_data:
                    del session_data[session_key]
                if session_key in USER_SESSIONS:
                    del USER_SESSIONS[session_key]
                session_key = email_lower
                session_obj = USER_SESSIONS[session_key]
                user_session = session_data[session_key]
            
            # Check if user exists
            user_data = sheets_service.get_user_by_email(email_lower)
            print(f"[USER DATA] Found: {user_data is not None}")

            if user_data:
                user_session["user_data"] = user_data
                user_session["client_id"] = user_data.get("client_id")
                user_session["stage"] = "password_verify"
                
                # Update session_obj with user data
                session_obj["normalized"]["email"] = email_lower
                session_obj["normalized"]["client_id"] = user_data.get("client_id")
                session_obj["normalized"]["room_alloted"] = user_data.get("Room Alloted", "")
                
                response = f"✅ Email found: *{incoming_msg}*\n\nPlease enter your password to continue."
            else:
                user_session["stage"] = "password_setup"
                response = (
                    f"👋 Welcome! We don't have an account for *{incoming_msg}* yet.\n\n"
                    "Let's create one! Please set a password (minimum 6 characters):"
                )
            
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
            reply_parts = response.split("\n\n")
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

        # Stage 2: Password Verification
        elif stage == "password_verify":
            stored_password = user_session["user_data"].get("Password")
            input_hash = hash_password(incoming_msg)
            print(f'[PASSWORD] Verifying for {session_key}')
            
            password_match = (stored_password == incoming_msg or stored_password == input_hash)
            
            if password_match:
                user_session["authenticated"] = True
                user_session["attempts"] = 0
                
                user_data = user_session["user_data"]
                workflow_stage = user_data.get("Workfow Stage", "").lower()
                booking_id = user_data.get("Booking Id", "")
                room_alloted = user_data.get("Room Alloted", "")
                checkin_date = user_data.get("CheckIn", "")
                checkout_date = user_data.get("Check Out", "")
                name = user_data.get("Name", "")
                email = user_data.get("Email", "")

                # Update session_obj with complete user data
                session_obj["normalized"].update({
                    "email": email,
                    "name": name,
                    "client_id": user_data.get("Client Id"),
                    "room_alloted": room_alloted,
                    "booking_id": booking_id,
                    "checkin": checkin_date,
                    "checkout": checkout_date,
                    "workflow_stage": workflow_stage
                })

                # Determine user type
                if workflow_stage in ["id_verified", "checked_in", "confirmed"] or booking_id or room_alloted:
                    user_session["user_type"] = "guest"
                    user_session["stage"] = "guest_chat" if room_alloted else "non_guest_chat"
                    
                    response = (
                        f"🎉 Welcome back, *{name}*!\n\n"
                        f"✅ Status: *VERIFIED GUEST*\n"
                        f"🏕️ Room: {room_alloted if room_alloted else 'TBD'}\n"
                        f"📅 Check-in: {checkin_date if checkin_date else 'TBD'}\n"
                        f"📅 Check-out: {checkout_date if checkout_date else 'TBD'}\n"
                        f"🆔 Booking ID: {booking_id if booking_id else 'Pending'}\n\n"
                        "You have full access to all our services:\n"
                        "🛏️ Room service (24/7)\n"
                        "💆 Spa & wellness\n"
                        "🏊 Swimming pool\n"
                        "🏋️ Gym & yoga\n"
                        "🦁 Safari experiences\n"
                        "🍽️ Bush dinners & dining\n\n"
                        "How can I assist you today?"
                    )
                else:
                    user_session["user_type"] = "non-guest"
                    user_session["stage"] = "non_guest_chat"
                    response = (
                        f"✅ Welcome back, *{name}*!\n\n"
                        "You're currently marked as a *VISITOR*.\n\n"
                        "You can:\n"
                        "📋 Ask general questions about ILORA RETREATS\n"
                        "🏕️ Book a luxury tent stay\n"
                        "🍽️ Learn about our dining options\n"
                        "🦁 Explore safari experiences\n\n"
                        "How can I help you today?"
                    )
            else:
                user_session["attempts"] = user_session.get("attempts", 0) + 1
                if user_session["attempts"] >= 3:
                    response = "❌ Too many failed attempts. Please restart by sending any message."
                    session_data[session_key] = {"stage": "welcome"}
                else:
                    response = f"❌ Incorrect password. Attempt {user_session['attempts']}/3. Please try again."
            
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated" if password_match else "unauthenticated")
            reply_parts = response.split("\n\n")
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

        # Stage 3: Password Setup (New User)
        elif stage == "password_setup":
            if len(incoming_msg) < 6:
                response = "❌ Password must be at least 6 characters. Please try again."
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
            
            password_hash = hash_password(incoming_msg)
            user_session["stage"] = "name_input"
            user_session["password"] = incoming_msg
            user_session["password_hash"] = password_hash
            response = "🔒 Password set successfully!\n\nPlease provide your *full name*:"
            
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
            reply_parts = response.split("\n\n")
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

        # Stage 4: Name Input (New User)
        elif stage == "name_input":
            user_session["name"] = incoming_msg
            user_session["stage"] = "phone_input"
            response = "📱 Great! Now please provide your *phone number*:"
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
            reply_parts = response.split("\n\n")
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

        # Stage 5: Phone Input (New User)
        elif stage == "phone_input":
            user_session["phone"] = incoming_msg
            
            client_id = f"ILR{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4().hex[:6]).upper()}"
            
            new_user_data = {
                "Client Id": client_id,
                "Name": user_session["name"],
                "Email": user_session["email"],
                "Phone Number": incoming_msg,
                "Password": user_session["password"],
                "Booking Id": "",
                "Workfow Stage": "Registered",
                "Room Alloted": "",
                "CheckIn": "",
                "Check Out": "",
                "Id Link": ""
            }
            
            success = sheets_service.create_new_user(new_user_data)
            
            if success:
                user_session["authenticated"] = True
                user_session["user_type"] = "non-guest"
                user_session["user_data"] = new_user_data
                user_session["client_id"] = client_id
                user_session["stage"] = "non_guest_chat"
                
                # Update session_obj
                session_obj["normalized"].update({
                    "email": user_session["email"],
                    "name": user_session["name"],
                    "client_id": client_id,
                    "phone": incoming_msg
                })
                
                response = (
                    f"✅ *Registration Complete!*\n\n"
                    f"🆔 Client ID: *{client_id}*\n"
                    f"Welcome to ILORA RETREATS, *{user_session['name']}*!\n\n"
                    "You can now:\n"
                    "📋 Ask questions about our retreat\n"
                    "🏕️ Book a luxury tent\n"
                    "🍽️ Explore our dining options\n"
                    "🦁 Learn about safari experiences\n\n"
                    "How can I assist you today?"
                )
            else:
                response = "⚠️ Registration failed. Please try again later."
                session_data[session_key] = {"stage": "welcome"}
            
            bot_reply_text = response
            log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated" if success else "unauthenticated")
            reply_parts = response.split("\n\n")
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
        
        # ==================== CHAT & BOOKING FLOW ====================
        
        # Check if user has room allocated (in-room guest)
        user_data = user_session.get("user_data", {})
        room_alloted = user_data.get('Room Alloted', '')
        
        print(f"[ROOM CHECK] Room Alloted: {room_alloted}, Stage: {stage}")
        
        if room_alloted and room_alloted.strip() != '':
            # IN-ROOM GUEST - Use inroom_bot with full user_data
            print(f"[INROOM BOT] Processing for {session_key}")
            
            bot_reply_text = inroom_bot.ask(
                query=user_input,
                user_type="guest",
                user_session=session_obj,
                session_key=session_key,
                user_data=user_data  # Pass complete user data
            )

            normalized = session_obj.get("normalized", {})
            guest_email = normalized.get("email")
            room_no = normalized.get("room_alloted")

            # Menu processing
            AVAILABLE_EXTRAS = {}
            EXTRAS_PRICE_BY_KEY = {}
            for c in menu[:]:
                if c.get("Type") == "Complimentary":
                    continue
                label = c.get("Item") or ""
                key = c.get("Item")
                try:
                    price = float(c.get("Price") or 0)
                except Exception:
                    price = 0.0
                if label:
                    AVAILABLE_EXTRAS[label] = key
                if key:
                    EXTRAS_PRICE_BY_KEY[key] = price

            message_lower = user_input.lower()
            addon_matches = [k for k in AVAILABLE_EXTRAS if k.lower() in message_lower]

            for price_addon in addon_matches:
                pending_balance += EXTRAS_PRICE_BY_KEY.get(AVAILABLE_EXTRAS[price_addon], 0)

            # Ticket creation
            created_ticket_id: Optional[str] = None
            try:
                if is_ticket_request(user_input, intent, addon_matches):
                    ticket_row = create_ticket_row_payload(user_input, guest_email)
                    try:
                        resp_json = push_row_to_sheet(TICKET_SHEET_NAME, ticket_row)
                        created_ticket_id = ticket_row.get("Ticket ID")
                        logger.info(f"Ticket created: {created_ticket_id}")
                        
                        try:
                            await broker.broadcast("ticket_created", {
                                "ticket_id": created_ticket_id,
                                "guest_email": guest_email,
                                "room_no": room_no,
                                "category": ticket_row.get("Category"),
                                "assigned_to": ticket_row.get("Assigned To"),
                                "status": ticket_row.get("Status"),
                                "created_at": ticket_row.get("Created At"),
                                "notes": ticket_row.get("Notes"),
                            })
                        except Exception as e:
                            logger.warning(f"Failed to broadcast ticket creation: {e}")
                    except Exception as e:
                        logger.warning(f"Failed to push ticket to sheet: {e}")
            except Exception as e:
                logger.warning(f"Ticket subsystem error: {e}")

            # Logging
            log_chat("web", session_key, user_input, bot_reply_text, intent, True)

            try:
                log_row = create_guest_log_row(session_key, guest_email, user_input, bot_reply_text, intent, True, created_ticket_id)
                try:
                    resp_log = push_row_to_sheet(GUEST_LOG_SHEET_NAME, log_row)
                    logger.info(f"Guest interaction logged: {log_row.get('Log ID')}")
                    
                    try:
                        await broker.broadcast("guest_log_created", {
                            "log_id": log_row.get("Log ID"),
                            "session_id": log_row.get("Session ID"),
                            "guest_email": guest_email,
                            "intent": intent,
                            "ticket_ref": created_ticket_id,
                            "timestamp": log_row.get("Timestamp")
                        })
                    except Exception as e:
                        logger.warning(f"Failed to broadcast guest log: {e}")
                except Exception as e:
                    logger.warning(f"Failed to push guest log to sheet: {e}")
            except Exception as e:
                logger.warning(f"Guest log subsystem error: {e}")

            reply_parts = bot_reply_text.split("\n\n") if isinstance(bot_reply_text, str) else [str(bot_reply_text)]
            return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
        
        else:
            # NOT IN-ROOM - Handle authentication and booking flow
            
            # Check authentication
            if not user_session.get("authenticated", False):
                response = "⚠️ Session expired. Please restart by sending any message."
                session_data[session_key] = {"stage": "welcome"}
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "unauthenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            user_type = user_session.get("user_type", "non-guest")
            user_identifier = user_session.get("email")

            # Non-Guest Chat
            if stage == "non_guest_chat":
                intent = classify_intent(incoming_msg.lower())
                logger.info(f"[NON-GUEST] Intent: {intent} for {session_key}")
                
                is_guest_service = any(service in incoming_msg.lower() for service in GUEST_ONLY_SERVICES)
                
                if is_guest_service and intent != "payment_request":
                    response = (
                        "🔒 This service is exclusive to our guests.\n\n"
                        "Would you like to book a stay with us? Reply *book* to see available tents!"
                    )
                elif intent == "payment_request" or "book" in incoming_msg.lower():
                    user_session["stage"] = "show_property"
                    response = "🌿 Let me show you our beautiful retreat..."
                else:
                    try:
                        answer = outroom_bot.ask(
                            incoming_msg,
                            user_type="non-guest",
                            user_session=user_identifier,
                            session_key=user_identifier,
                            user_data=user_data  # Pass user data
                        )
                        response = f"💬 {answer}"
                    except Exception as e:
                        logger.error(f"Bot error: {e}")
                        response = "⚠️ I'm having trouble processing that. Please try again."
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Show Property
            elif stage == "show_property":
                property_images = getattr(Config, 'PROPERTY_IMAGES', [])
                
                text_response_part1 = "🏕️ *ILORA RETREATS - Luxury Safari Experience*"
                
                available_tents = sheets_service.get_available_tents()
                
                if available_tents > 0:
                    user_session["stage"] = "booking_nights"
                    text_response_part2 = (
                        f"✨ We have *{available_tents} luxury tents* available out of {TOTAL_TENTS}!\n\n"
                        f"💰 *Rate:* ₹{ROOM_PRICES['Luxury Tent']:,}/night\n"
                        f"(Approximately USD 500-650)\n\n"
                        "✅ *Includes:*\n"
                        "🛏️ Fully equipped tent with en-suite bathroom\n"
                        "🌅 Private veranda\n"
                        "🍽️ Full-board dining (breakfast, lunch, dinner)\n"
                        "🏊 Pool, spa & gym access\n"
                        "🧘 Yoga sessions\n\n"
                        "*How many nights* would you like to stay?\n"
                        "Reply with a number (e.g., 3)"
                    )
                else:
                    user_session["stage"] = "non_guest_chat"
                    text_response_part2 = (
                        "😔 We're currently fully booked!\n\n"
                        "📧 Please contact us at reservations@iloraretreats.com\n"
                        "📞 Or call us for future availability."
                    )

                final_reply_text = f"{text_response_part1}\n\n{text_response_part2}"
                final_reply_parts = [text_response_part1, text_response_part2]
                images_to_send = property_images[:6] if property_images else None

                log_chat("WhatsApp", user_number, incoming_msg, final_reply_text, "authenticated")

                return ChatResp(
                    reply=final_reply_text,
                    reply_parts=final_reply_parts,
                    intent=intent,
                    actions=actions,
                    media_urls=images_to_send
                )

            # Booking: Number of Nights
            elif stage == "booking_nights":
                try:
                    nights = int(incoming_msg)
                    if nights <= 0 or nights > 30:
                        response = "❌ Please enter a valid number between 1 and 30 nights."
                    else:
                        user_session["nights"] = nights
                        user_session["stage"] = "booking_checkin"
                        
                        total = ROOM_PRICES["Luxury Tent"] * nights
                        user_session["total_amount"] = total
                        
                        response = (
                            f"🌙 *{nights} night(s)* - Excellent choice!\n"
                            f"💰 Estimated Total: ₹{total:,}\n\n"
                            "📅 When would you like to *check in*?\n"
                            "Please provide the date in format: *DD-MM-YYYY*\n"
                            "(e.g., 15-12-2025)"
                        )
                except ValueError:
                    response = "❌ Please enter a valid number of nights (e.g., 2, 3, 5)."
                    try:
                        fallback = outroom_bot.ask(incoming_msg, user_type="non-guest", user_data=user_data)
                        response += f"\n\n{fallback}"
                    except:
                        pass
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Booking: Check-in Date
            elif stage == "booking_checkin":
                if not validate_date(incoming_msg):
                    response = "❌ Invalid date format. Please use DD-MM-YYYY (e.g., 15-12-2025)."
                    try:
                        fallback = outroom_bot.ask(incoming_msg, user_type="non-guest", user_data=user_data)
                        response += f"\n\n{fallback}"
                    except:
                        pass
                    user_session["stage"] = "non_guest_chat"
                else:
                    try:
                        checkin_date = datetime.strptime(incoming_msg, "%d-%m-%Y")
                        today = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
                        
                        if checkin_date < today:
                            response = "❌ Check-in date cannot be in the past. Please enter a future date."
                        else:
                            nights = user_session["nights"]
                            checkout_date = checkin_date + timedelta(days=nights)
                            
                            user_session["checkin_date"] = incoming_msg
                            user_session["checkout_date"] = checkout_date.strftime("%d-%m-%Y")
                            user_session["stage"] = "booking_payment"
                            
                            total = user_session["total_amount"]
                            
                            response = (
                                "💳 *Payment Method*\n\n"
                                f"📋 *Booking Summary:*\n"
                                f"👤 Name: {user_session['name']}\n"
                                f"🏕️ Room: Luxury Tent\n"
                                f"📅 Check-in: {incoming_msg}\n"
                                f"📅 Check-out: {user_session['checkout_date']}\n"
                                f"🌙 Nights: {nights}\n"
                                f"💰 Total: ₹{total:,}\n\n"
                                "How would you like to pay?\n"
                                "1️⃣ Online Payment (Secure)\n"
                                "2️⃣ Pay on Arrival\n\n"
                                "Reply with *1* or *2*"
                            )
                    except Exception as e:
                        logger.error(f"Date processing error: {e}")
                        response = "❌ Error processing date. Please try again with format DD-MM-YYYY"
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Booking: Payment Method
            elif stage == "booking_payment":
                if incoming_msg not in ["1", "2"]:
                    response = "❌ Please select 1 for Online Payment or 2 for Pay on Arrival."
                    try:
                        fallback = outroom_bot.ask(incoming_msg, user_type="non-guest", user_data=user_data)
                        response += f"\n\n{fallback}"
                    except:
                        pass
                    user_session["stage"] = "non_guest_chat"
                else:
                    payment_mode = "Online" if incoming_msg == "1" else "Cash on Arrival"
                    user_session["payment_mode"] = payment_mode
                    user_session["stage"] = "booking_confirm"
                    
                    response = (
                        "✅ *Please confirm your booking:*\n\n"
                        f"👤 Name: {user_session['name']}\n"
                        f"📧 Email: {user_session['email']}\n"
                        f"📱 Phone: {user_session['phone']}\n"
                        f"🏕️ Room: Luxury Tent\n"
                        f"📅 Check-in: {user_session['checkin_date']}\n"
                        f"📅 Check-out: {user_session['checkout_date']}\n"
                        f"🌙 Nights: {user_session['nights']}\n"
                        f"💳 Payment: {payment_mode}\n"
                        f"💰 Total: ₹{user_session['total_amount']:,}\n\n"
                        "Reply *YES* to confirm or *NO* to cancel."
                    )
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Booking: Confirmation
            elif stage == "booking_confirm":
                if incoming_msg.lower() == "yes":
                    try:
                        booking_id = f"ILORA{datetime.now().strftime('%Y%m%d')}{str(uuid.uuid4().hex[:6]).upper()}"
                        
                        booking_data = {
                            "email": user_session["email"],
                            "booking_id": booking_id,
                            "workflow_stage": "booking_confirmed",
                            "room_alloted": "Luxury Tent",
                            "checkin": user_session["checkin_date"],
                            "checkout": user_session["checkout_date"]
                        }
                        
                        booking_success = sheets_service.update_booking(booking_data)
                        
                        if booking_success:
                            # Update user_session with booking data
                            user_session["booking_id"] = booking_id
                            if "user_data" not in user_session:
                                user_session["user_data"] = {}
                            user_session["user_data"]["Booking Id"] = booking_id
                            user_session["user_data"]["Room Alloted"] = "Luxury Tent"
                            user_session["user_data"]["CheckIn"] = user_session["checkin_date"]
                            user_session["user_data"]["Check Out"] = user_session["checkout_date"]
                            
                            # Update session_obj
                            session_obj["normalized"]["booking_id"] = booking_id
                            session_obj["normalized"]["room_alloted"] = "Luxury Tent"
                            session_obj["normalized"]["checkin"] = user_session["checkin_date"]
                            session_obj["normalized"]["checkout"] = user_session["checkout_date"]
                            
                            if user_session["payment_mode"] == "Online":
                                pay_url = create_checkout_session(
                                    session_id=booking_id,
                                    room_type="Luxury Tent",
                                    nights=user_session["nights"],
                                    cash=False
                                )
                                
                                if pay_url:
                                    response = (
                                        "🎉 *Booking Confirmed!*\n\n"
                                        f"🆔 Booking ID: *{booking_id}*\n"
                                        f"👤 Name: {user_session['name']}\n"
                                        f"📧 Email: {user_session['email']}\n\n"
                                        "💳 *Complete your payment here:*\n"
                                        f"{pay_url}\n\n"
                                        "After payment, your status will be updated to *VERIFIED GUEST* "
                                        "and you'll have full access to all services!\n\n"
                                        "📧 A confirmation email has been sent to your inbox.\n\n"
                                        "How would you like to do the checkin?\n"
                                        "1️⃣ Web Checkin (Secure)\n"
                                        "2️⃣ CheckIn on Arrival\n\n"
                                        "Reply with *1* or *2*"
                                    )
                                    user_session["stage"] = "checkin_method"
                                else:
                                    response = (
                                        "🎉 *Booking Confirmed!*\n\n"
                                        f"🆔 Booking ID: *{booking_id}*\n\n"
                                        "⚠️ Payment link generation failed.\n"
                                        "Please contact us at reservations@iloraretreat.com"
                                    )
                                    user_session["stage"] = "non_guest_chat"
                            else:
                                response = (
                                    "🎉 *Booking Confirmed!*\n\n"
                                    f"🆔 Booking ID: *{booking_id}*\n"
                                    f"👤 Name: {user_session['name']}\n"
                                    f"📧 Email: {user_session['email']}\n"
                                    f"📅 Check-in: {user_session['checkin_date']}\n"
                                    f"📅 Check-out: {user_session['checkout_date']}\n"
                                    f"💰 Total: ₹{user_session['total_amount']:,}\n\n"
                                    "💵 Payment will be collected on arrival.\n\n"
                                    "We look forward to welcoming you to ILORA RETREATS! 🌿\n\n"
                                    "📧 A confirmation email has been sent.\n\n"
                                    "How would you like to do the checkin?\n"
                                    "1️⃣ Web Checkin (Secure)\n"
                                    "2️⃣ CheckIn on Arrival\n\n"
                                    "Reply with *1* or *2*"
                                )
                                user_session["stage"] = "checkin_method"
                            
                            if user_session["payment_mode"] == "Cash on Arrival":
                                sheets_service.update_workflow_stage(user_session["email"], "booked")
                        else:
                            response = "⚠️ Booking failed. Please try again or contact support."
                            user_session["stage"] = "non_guest_chat"
                    except Exception as e:
                        logger.error(f"Booking error: {e}")
                        response = "⚠️ An error occurred during booking. Please try again."
                        user_session["stage"] = "non_guest_chat"
                else:
                    response = "❌ Booking cancelled. How else can I help you?"
                    user_session["stage"] = "non_guest_chat"
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
                
            # Check-in Method Selection
            elif stage == "checkin_method":
                if incoming_msg == "1":
                    checkin_url = "https://forms.gle/RvnsymRmBoKu3Ns26"
                    response = (
                        f"🔒 You have selected Web Checkin (Secure).\n\n"
                        f"Please follow the link to complete your checkin:\n{checkin_url}\n\n"
                        "Once completed, you'll have full access to all services!"
                    )
                    user_session["stage"] = "non_guest_chat"
                elif incoming_msg == "2":
                    response = (
                        "🚶 You have selected CheckIn on Arrival.\n\n"
                        "Please proceed to the reception upon arrival with your booking ID.\n\n"
                        "Is there anything else I can help you with?"
                    )
                    user_session["stage"] = "non_guest_chat"
                else:
                    response = "❓ Invalid option. Please reply with *1* for Web Checkin or *2* for CheckIn on Arrival."
                    try:
                        fallback = outroom_bot.ask(incoming_msg, user_type="non-guest", user_data=user_data)
                        response += f"\n\n{fallback}"
                    except:
                        pass
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Guest Chat (Verified Guests without room allocation yet)
            elif stage == "guest_chat":
                intent = classify_intent(incoming_msg.lower())
                logger.info(f"[GUEST CHAT] Intent: {intent} for {session_key}")

                # Handle add-on bookings
                if intent.startswith("book_addon"):
                    matches = [key for key in ADDON_MAPPING if key in incoming_msg.lower()]
                    if matches:
                        try:
                            extras = list(set(ADDON_MAPPING[m] for m in matches))
                            session_id = user_session.get("client_id", str(uuid.uuid4()))
                            pay_url = create_addon_checkout_session(session_id=session_id, extras=extras)
                            
                            if pay_url:
                                addon_names = ', '.join([e.replace('_', ' ').title() for e in extras])
                                response = (
                                    f"🎯 *Add-on Booking*\n\n"
                                    f"📋 Selected: {addon_names}\n"
                                    f"🆔 Booking ID: {user_session.get('booking_id', 'N/A')}\n\n"
                                    f"Complete payment here:\n{pay_url}"
                                )
                            else:
                                response = "⚠️ Could not generate payment link. Please contact our concierge."
                        except Exception as e:
                            logger.error(f"Add-on error: {e}")
                            response = "⚠️ Error processing add-on. Please try again or contact concierge."
                    else:
                        response = (
                            "❓ Which add-on would you like to book?\n\n"
                            "Available options:\n"
                            "🧖 Spa & Massage\n"
                            "🎈 Hot Air Balloon Ride\n"
                            "🦁 Game Drive\n"
                            "🚶 Walking Safari\n"
                            "🍽️ Bush Dinner\n"
                            "⭐ Stargazing Experience\n"
                            "🎭 Maasai Cultural Experience"
                        )
                else:
                    # General guest query
                    try:
                        answer = outroom_bot.ask(
                            incoming_msg,
                            user_type="guest",
                            user_session=user_identifier,
                            session_key=user_identifier,
                            user_data=user_data  # Pass user data
                        )
                        response = f"💬 {answer}"
                    except Exception as e:
                        logger.error(f"Bot error: {e}")
                        response = "⚠️ I'm having trouble with that. Let me connect you with our concierge team."
                
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "authenticated")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

            # Default fallback
            else:
                response = "⚠️ Something went wrong. Please restart by sending any message."
                session_data[session_key] = {"stage": "welcome"}
                bot_reply_text = response
                log_chat("WhatsApp", user_number, incoming_msg, response, "error")
                reply_parts = response.split("\n\n")
                return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)
        

    except Exception as e:
        logger.error(f"[CRITICAL ERROR] in /chat endpoint: {str(e)}", exc_info=True)
        bot_reply_text = 'ERR!! Could not reach the server. Please try again.'
        intent = None
        actions = ChatActions()
        reply_parts = [bot_reply_text]
        return ChatResp(reply=bot_reply_text, reply_parts=reply_parts, intent=intent, actions=actions)

# ------------------------- Run locally -------------------------
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("unified_backend:app", host="0.0.0.0", port=int(os.environ.get("PORT", 8000)), reload=True)
    
