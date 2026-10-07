from flask import Flask, render_template, request, jsonify
from pathlib import Path
import re
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification


# =========================================================
# SatyaCheck - MuRIL Fake News Detector
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_DIR = BASE_DIR / "model" / "muril_fake_news"

MODEL_NAME = "MuRIL Fine-Tuned Fake News Classifier"

MAX_LENGTH = 128
DECISION_THRESHOLD = 0.60

app = Flask(__name__)

tokenizer = None
model = None


# =========================================================
# Load MuRIL model
# =========================================================

def load_model():
    global tokenizer, model

    if not MODEL_DIR.exists():
        raise FileNotFoundError(
            f"MuRIL model not found at: {MODEL_DIR}"
        )

    print("=" * 60)
    print("Loading SatyaCheck MuRIL model...")
    print("=" * 60)

    tokenizer = AutoTokenizer.from_pretrained(
        str(MODEL_DIR)
    )

    model = AutoModelForSequenceClassification.from_pretrained(
        str(MODEL_DIR)
    )

    model.eval()

    print("MuRIL model loaded successfully.")
    print(f"Model path: {MODEL_DIR}")
    print("=" * 60)


def ensure_model():
    if model is None or tokenizer is None:
        load_model()


# =========================================================
# Detect suspicious patterns
# =========================================================

def detect_risk_signals(title, article):

    text = f"{title} {article}".lower()

    signals = []

    free_patterns = [
        "free",
        "muft",
        "मुफ्त",
        "free recharge",
        "free internet",
        "free mobile",
        "free gift",
        "free money",
        "cash prize",
        "reward",
        "इनाम",
    ]

    urgency_patterns = [
        "limited time",
        "limited offer",
        "jaldi karein",
        "abhi karein",
        "claim now",
        "claim karein",
        "turant",
        "immediately",
        "hurry",
        "last chance",
    ]

    action_patterns = [
        "click",
        "click here",
        "link par click",
        "link par",
        "open link",
        "claim",
        "register",
        "registration",
        "form bharein",
        "apply now",
    ]

    government_patterns = [
        "modi govt",
        "modi government",
        "government is giving",
        "sarkar de rahi",
        "sarkar degi",
        "government free",
        "govt free",
        "bharat sarkar",
        "indian government",
    ]

    scam_patterns = [
        "recharge",
        "otp",
        "bank account",
        "upi",
        "cashback",
        "prize",
        "lottery",
        "voucher",
        "coupon",
        "gift card",
    ]

    def contains_any(patterns):
        return any(pattern in text for pattern in patterns)

    if contains_any(free_patterns):
        signals.append("free/reward claim")

    if contains_any(urgency_patterns):
        signals.append("urgency language")

    if contains_any(action_patterns):
        signals.append("call-to-action language")

    if contains_any(government_patterns):
        signals.append("government-benefit claim")

    if contains_any(scam_patterns):
        signals.append("recharge/scam-related language")

    urls = re.findall(
        r"https?://\S+|www\.\S+",
        text,
        flags=re.I
    )

    if urls:
        signals.append("external link")

    suspicious_domains = [
        "fakelink",
        "bit.ly",
        "tinyurl",
        "t.co",
    ]

    if any(
        domain in url
        for url in urls
        for domain in suspicious_domains
    ):
        signals.append("suspicious link pattern")

    return signals


# =========================================================
# Predict news
# =========================================================

def predict_news(title, article):

    ensure_model()

    title = str(title or "").strip()
    article = str(article or "").strip()

    # EXACT SAME FORMAT USED DURING TRAINING
    combined_text = (
        title
        + " [SEP] "
        + article
    )

    inputs = tokenizer(
        combined_text,
        truncation=True,
        max_length=MAX_LENGTH,
        padding=True,
        return_tensors="pt",
    )

    with torch.no_grad():

        outputs = model(**inputs)

        probabilities = torch.softmax(
            outputs.logits,
            dim=-1
        )[0]

    prediction = int(
        torch.argmax(probabilities).item()
    )

    confidence = float(
        probabilities[prediction].item()
    )

    # Additional risk analysis
    risk_signals = detect_risk_signals(
        title,
        article
    )

    risk_score = len(risk_signals)

    # Final classification
    if confidence < DECISION_THRESHOLD:

        label = "UNCERTAIN"
        is_real = None

    elif risk_score >= 3:

        label = "LIKELY FAKE NEWS"
        is_real = False

    elif prediction == 1:

        label = "REAL NEWS"
        is_real = True

    else:

        label = "LIKELY FAKE NEWS"
        is_real = False

    return {
        "label": label,
        "is_real": is_real,
        "confidence": round(
            confidence * 100,
            2
        ),
        "threshold": round(
            DECISION_THRESHOLD * 100,
            2
        ),
        "risk_score": risk_score,
        "risk_signals": risk_signals,
        "model": MODEL_NAME,
        "warning": (
            "This is an ML and risk-pattern assessment, "
            "not proof of truth. Verify important claims "
            "with reliable primary sources."
        ),
    }


# =========================================================
# Home page
# =========================================================

@app.get("/")
def home():

    return render_template(
        "index.html",
        result=None,
        title="",
        article="",
        error=None,
        metrics={
            "model": MODEL_NAME
        }
    )


# =========================================================
# Web form
# =========================================================

@app.post("/")
def analyze():

    title = request.form.get(
        "title",
        ""
    ).strip()

    article = request.form.get(
        "text",
        ""
    ).strip()

    result = None
    error = None

    if len(title) < 5:

        error = (
            "Please enter a headline "
            "of at least 5 characters."
        )

    elif len(article) < 30:

        error = (
            "Please enter at least 30 "
            "characters of article/message text."
        )

    else:

        try:
            result = predict_news(
                title,
                article
            )

        except Exception as exc:
            error = f"Prediction error: {exc}"

    return render_template(
        "index.html",
        result=result,
        title=title,
        article=article,
        error=error,
        metrics={
            "model": MODEL_NAME
        }
    )


# =========================================================
# API
# =========================================================

@app.post("/api/predict")
def api_predict():

    data = request.get_json(
        silent=True
    ) or {}

    title = str(
        data.get("title", "")
    ).strip()

    article = str(
        data.get("text", "")
    ).strip()

    if len(title) < 5:

        return jsonify({
            "error": "Title must contain at least 5 characters."
        }), 400

    if len(article) < 30:

        return jsonify({
            "error": "Text must contain at least 30 characters."
        }), 400

    try:

        return jsonify(
            predict_news(
                title,
                article
            )
        )

    except Exception as exc:

        return jsonify({
            "error": str(exc)
        }), 503


# =========================================================
# Health check
# =========================================================

@app.get("/health")
def health():

    try:

        ensure_model()

        return jsonify({
            "status": "ok",
            "model_loaded": True,
            "model": MODEL_NAME
        })

    except Exception as exc:

        return jsonify({
            "status": "error",
            "model_loaded": False,
            "error": str(exc)
        }), 503


# =========================================================
# Start
# =========================================================

if __name__ == "__main__":

    try:
        load_model()

    except Exception as exc:
        print("\nERROR loading model:")
        print(exc)

    app.run(
        host="0.0.0.0",
        port=5000,
        debug=False
    )
