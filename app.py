import io, json, re
import streamlit as st
import pandas as pd
import pdfplumber
from pydantic import BaseModel
from typing import List, Optional, Any, Dict
import requests

st.set_page_config(page_title="Pension PDF Answers", layout="wide")
APP_TITLE = "Pension PDF Answers"
CONF_THRESHOLD = 75
MAX_PAGES = 40

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY", "")
OPENAI_MODEL = st.secrets.get("OPENAI_MODEL", "gpt-4o-mini")
OPENAI_URL = "https://api.openai.com/v1/chat/completions"

class Evidence(BaseModel):
    page: Optional[int] = None
    snippet: Optional[str] = None
    source: str = "baseline"

class Answer(BaseModel):
    key: str
    label: str
    type: str
    value: Optional[Any] = None
    confidence: int = 0
    evidence: Optional[Evidence] = None

def read_questions() -> List[Dict[str, Any]]:
    return json.loads(open("questions.json","r",encoding="utf-8").read())

def clean_text(s: str) -> str:
    s = re.sub(r"[ \t]+"," ", s).replace("\u00a0"," ")
    return s.strip()

def extract_pdf_text(file) -> List[str]:
    pages=[]
    with pdfplumber.open(file) as pdf:
        for i, page in enumerate(pdf.pages[:MAX_PAGES]):
            txt = page.extract_text() or ""
            pages.append(clean_text(txt))
    return pages

def baseline_rules(pages: List[str], questions: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    res: Dict[str, Dict[str, Any]] = {}
    text_all = "\n".join(pages)

    def find(patterns, text):
        for p in patterns:
            m = re.search(p, text, flags=re.I)
            if m:
                val = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                span = m.span()
                snippet = text[max(0, span[0]-80):min(len(text), span[1]+80)]
                return val, snippet
        return None, None

    # Money like £12,345.67 or 12345.67 or GBP 1,234
    MONEY = r"(?:GBP|£)?\s*([0-9]{1,3}(?:,[0-9]{3})*(?:\.[0-9]{2})?|[0-9]+(?:\.[0-9]{2})?)"
    # Dates: 12/03/2024, 12-03-24, 12 March 2024, 12 Mar 2024
    DATE = r"((?:[0-3]?\d[\/\-][0-1]?\d[\/\-](?:\d{2}|\d{4}))|(?:[0-3]?\d\s+(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Sept|Oct|Nov|Dec|January|February|March|April|June|July|August|September|October|November|December)\s+\d{2,4}))"

    patterns = {
        "plan_number": [
            r"(?:policy|plan|account|reference|policy\s*no\.?|plan\s*no\.?)\s*[:#]?\s*([A-Z0-9\-\/]{6,})",
            r"\b([A-Z]{2,3}\d{5,}|[A-Z0-9]{8,})\b"
        ],
        "pstr_number":  [
            r"\bPSTR[^A-Za-z0-9]*([0-9]{8})\b",
            r"\b(00\d{6})\b"
        ],
        "selected_retirement_age": [r"(?:selected|normal)\s+retirement\s+(?:age|date)[^\d]*(\d{2})"],
        "plan_status": [r"\b(in\s*force|paid\s*up|lapsed)\b"],

        "current_value": [rf"(?:current|fund)\s+value[^\d£]*{MONEY}"],
        "current_value_date": [rf"(?:as at|valuation date|valued on)[^\dA-Z]*{DATE}"],

        "transfer_value": [rf"(?:cash\s+equivalent\s+transfer\s+value|transfer\s+value)[^\d£]*{MONEY}"],
        "transfer_value_date": [rf"(?:as at|valuation date|valued on)[^\dA-Z]*{DATE}"],

        "life_cover": [r"life\s*cover\s*[: ]+(yes|no)"],
        "protected_retirement_age": [r"protected\s+retirement\s+age\s*[: ]+(yes|no)"],
        "pension_sharing_earmarking": [r"(pension\s+sharing|earmarking)\s*[: ]+(yes|no)"],
        "advice_available": [r"advice\s+available\s*[: ]+(yes|no)"],
        "advice_cost": [rf"advice\s+(?:charge|cost)[^\d£]*{MONEY}"],
        "tax_free_cash_amount": [rf"(?:tax[-\s]?free\s+cash|PCLS)[^\d£]*{MONEY}"],
        "partial_transfers_allowed": [r"partial\s+transfers?\s*[: ]+(yes|no)"],
        "funds_available_count": [r"(?:funds\s+available|available\s+funds)[^\d]*(\d{1,4})"],
        "funds_max_hold": [r"maximum\s+number\s+of\s+funds[^\d]*(\d{1,3})"]
    }

    def norm_money(x):
        x = re.sub(r"[£,\s]|GBP", "", x, flags=re.I)
        try:
            return float(x)
        except:
            return None

    for q in questions:
        key = q["key"]
        value, snippet = (None, None)
        if key in patterns:
            value, snippet = find(patterns[key], text_all)

        conf = 0
        if value is not None:
            t = q["type"]
            v = value
            if t in ["currency","number"]:
                v = norm_money(v) if t == "currency" else (float(v) if re.fullmatch(r"\d+(?:\.\d+)?", str(v)) else None)
                conf = 75 if v is not None else 60
            elif t == "boolean":
                sv = str(value).strip().lower()
                if sv in ["yes","true","y"]: v, conf = True, 80
                elif sv in ["no","false","n"]: v, conf = False, 80
                else: v, conf = None, 0
            else:
                conf = 80
            res[key] = {"value": v, "confidence": conf, "evidence": {"page": None, "snippet": snippet, "source":"baseline"}}
        else:
            res[key] = {"value": None, "confidence": 0, "evidence": None}
    return res

def llm_fill(pages: List[str], questions: List[Dict[str, Any]]) -> Dict[str, Any]:
    if not OPENAI_API_KEY:
        return {}
    text = "\n\n".join(pages)[:12000]
    keys = [q["key"] for q in questions]
    hints = {q["key"]: q.get("hint","") for q in questions}
    sys = "You extract answers from pension documents. Return only valid JSON with exactly the requested keys and null for unknown."
    prompt = f"""Keys: {json.dumps(keys)}
Hints: {json.dumps(hints, ensure_ascii=False, indent=2)}
Rules:
- Use numbers for money, no currency symbols.
- true/false for yes/no.
- Use ISO-like dates if possible; otherwise DD/MM/YYYY as seen.
- If not certain, return null.
Source:
{text}
"""
    headers = {"Authorization": f"Bearer {OPENAI_API_KEY}"}
    body = {
        "model": OPENAI_MODEL,
        "temperature": 0.0,
        "response_format": {"type": "json_object"},
        "messages": [{"role":"system","content":sys},{"role":"user","content":prompt}],
    }
    try:
        r = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=body, timeout=60)
        r.raise_for_status()
        data = r.json()
        return json.loads(data["choices"][0]["message"]["content"])
    except Exception as e:
        st.warning(f"LLM extraction failed: {e}")
        return {}

def items_to_df(items: List[Answer]) -> pd.DataFrame:
    rows=[]
    for a in items:
        rows.append({"Key":a.key,"Question":a.label,"Answer":"" if a.value is None else a.value,
                     "Confidence":a.confidence,"Source":a.evidence.source if a.evidence else "",
                     "Snippet":a.evidence.snippet if a.evidence else ""})
    return pd.DataFrame(rows)

st.title(APP_TITLE)
st.caption("Upload a pension PDF. Answers and gaps will display below. Files are processed in memory and not stored.")

uploaded = st.file_uploader("Drop a PDF", type=["pdf"])
conf_threshold = st.slider("Low-confidence threshold", 0, 100, CONF_THRESHOLD)

if uploaded:
    questions = read_questions()
    pages = extract_pdf_text(uploaded)
    if not any(pages):
        st.error("No readable text found in the PDF. Please upload a text-based PDF.")
        st.stop()

    base = baseline_rules(pages, questions)
    needs_llm = any((v["value"] is None or v["confidence"] < 70) for v in base.values())
    llm_data = llm_fill(pages, questions) if needs_llm else {}

    items: List[Answer] = []
    for q in questions:
        key, label, t = q["key"], q["label"], q["type"]
        b = base.get(key, {})
        b_val, b_conf, b_ev = b.get("value"), b.get("confidence", 0), b.get("evidence")
        l_val = llm_data.get(key)

        def norm(val):
            if val is None: return None
            try:
                if t == "currency": return float(str(val).replace(",",""))
                if t == "number": return float(val)
                if t == "boolean": return str(val).lower() in ["true","yes","y","1"]
                return val
            except: return val

        l_val = norm(l_val)

        if b_val is not None and b_conf >= 80:
            final, conf, src, ev = b_val, b_conf, "baseline", Evidence(**b_ev) if b_ev else None
        elif l_val is not None:
            final, conf, src, ev = l_val, max(70, b_conf), "llm", Evidence(page=None, snippet=None, source="llm")
        else:
            final, conf, src, ev = None, b_conf, "baseline", Evidence(**b_ev) if b_ev else None

        items.append(Answer(key=key, label=label, type=t, value=final, confidence=int(conf), evidence=ev))

    df = items_to_df(items)
    low_mask = (df["Answer"] == "") | (df["Confidence"] < conf_threshold)
    if low_mask.any():
        st.info(f"{int(low_mask.sum())} fields need attention. Edit them below before export.")

    edited = st.data_editor(df, hide_index=True, use_container_width=True, disabled=["Key","Question","Source","Snippet"])

    csv_bytes = edited.to_csv(index=False).encode("utf-8")
    from io import BytesIO
    bio = BytesIO()
    edited.to_excel(bio, index=False)

    c1, c2 = st.columns(2)
    with c1:
        st.download_button("Download CSV", csv_bytes, file_name="answers.csv", mime="text/csv")
    with c2:
        st.download_button("Download Excel", bio.getvalue(), file_name="answers.xlsx",
                           mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
