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

    def find(patterns):
        for p in patterns:
            m = re.search(p, text_all, flags=re.I)
            if m:
                val = m.group(2) if m.lastindex and m.lastindex >= 2 else m.group(1)
                span = m.span()
                snippet = text_all[max(0, span[0]-60):min(len(text_all), span[1]+60)]
                return val, snippet
        return None, None

    patterns = {
        "plan_number": [r"(policy|plan|account|reference)[^:\n]*[:# ]+([A-Z0-9\-\/]{6,})", r"\b([A-Z0-9]{8,})\b"],
        "pstr_number":  [r"\bPSTR[^A-Za-z0-9]*([0-9]{8})\b", r"\b(00\d{6})\b"],
        "selected_retirement_age": [r"(selected|normal) retirement (age|date)[^\d]*(\d{2})"],
        "plan_status": [r"\b(in force|paid up|lapsed)\b"],
        "current_value": [r"(current|fund) value[^\d£]*([£$]?[0-9,]+\.\d{2}|[£$]?[0-9,]+)"],
        "current_value_date": [r"(as at|valuation date)[^\d]*(\d{1,2}[\/\-][0-1]?\d[\/\-](?:\d{2}|\d{4}))"],
        "transfer_value": [r"transfer value[^\d£]*([£$]?[0-9,]+\.\d{2}|[£$]?[0-9,]+)"],
        "transfer_value_date": [r"(as at|valuation date)[^\d]*(\d{1,2}[\/\-][0-1]?\d[\/\-](?:\d{2}|\d{4}))"],
        "life_cover": [r"life cover[: ]+(yes|no)"],
        "protected_retirement_age": [r"protected retirement age[: ]+(yes|no)"],
        "pension_sharing_earmarking": [r"(pension sharing|earmarking)[: ]+(yes|no)"],
        "advice_available": [r"advice available[: ]+(yes|no)"],
        "advice_cost": [r"advice (charge|cost)[^\d£]*([£$]?[0-9,]+\.\d{2}|[£$]?[0-9,]+)"],
        "tax_free_cash_amount": [r"(tax[- ]?free cash|PCLS)[^\d£]*([£$]?[0-9,]+\.\d{2}|[£$]?[0-9,]+)"],
        "partial_transfers_allowed": [r"partial transfers?[: ]+(yes|no)"],
        "funds_available_count": [r"(funds available|available funds)[^\d]*(\d{1,4})"],
        "funds_max_hold": [r"maximum number of funds[^\d]*(\d{1,3})"]
    }

    for q in questions:
        key = q["key"]
        value, snippet = (None, None)
        if key in patterns:
            value, snippet = find(patterns[key])

        conf = 0
        if value is not None:
            t = q["type"]
            v = value
            if t in ["currency","number"]:
                v = re.sub(r"[£$,]","", v)
                try: v=float(v); conf=70
                except: v=value; conf=60
            elif t == "boolean":
                sv = str(value).strip().lower()
                if sv in ["yes","true","y"]: v, conf = True, 70
                elif sv in ["no","false","n"]: v, conf = False, 70
                else: v, conf = None, 0
            else:
                conf = 75
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
