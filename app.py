"""
Rising Waters – ML-Based Flood Prediction System
Flask Application
"""

import os
import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash

# ─── App Setup ────────────────────────────────────────────────────────────────
app = Flask(__name__)
app.secret_key = os.environ.get("SECRET_KEY", "risingwaters_secret_2024")

MODEL_PATH    = "floods.save"
SCALER_PATH   = "transform.save"
FEATURE_COLS  = ["Annual_Rainfall", "Seasonal_Rainfall", "Temperature",
                 "Cloud_Cover", "Humidity"]


# ─── Helpers ──────────────────────────────────────────────────────────────────

def load_artifacts():
    """Load model and scaler; raise informative errors if missing."""
    try:
        model  = joblib.load(MODEL_PATH)
        scaler = joblib.load(SCALER_PATH)
        return model, scaler
    except FileNotFoundError as e:
        raise RuntimeError(
            "Model files not found. Run train_model.py first."
        ) from e


def validate_input(form):
    """Parse and validate form inputs. Returns (data_dict, errors)."""
    errors = {}
    data   = {}

    bounds = {
        "Annual_Rainfall":   (0,    5000,  "mm"),
        "Seasonal_Rainfall": (0,    3000,  "mm"),
        "Temperature":       (-10,  60,    "°C"),
        "Cloud_Cover":       (0,    100,   "%"),
        "Humidity":          (0,    100,   "%"),
    }

    for field, (lo, hi, unit) in bounds.items():
        raw = form.get(field, "").strip()
        if not raw:
            errors[field] = f"{field.replace('_', ' ')} is required."
            continue
        try:
            val = float(raw)
        except ValueError:
            errors[field] = f"{field.replace('_', ' ')} must be a number."
            continue
        if not (lo <= val <= hi):
            errors[field] = f"{field.replace('_', ' ')} must be between {lo} and {hi} {unit}."
            continue
        data[field] = val

    return data, errors


# ─── Routes ───────────────────────────────────────────────────────────────────

@app.route("/")
def home():
    """Landing / hero page."""
    return render_template("home.html")


@app.route("/predict", methods=["GET"])
def predict():
    """Show prediction form."""
    return render_template("predict.html")


@app.route("/prediction", methods=["POST"])
def prediction():
    """Handle form submission → predict → show result."""
    data, errors = validate_input(request.form)

    if errors:
        for msg in errors.values():
            flash(msg, "danger")
        return redirect(url_for("predict"))

    try:
        model, scaler = load_artifacts()

        df_input = pd.DataFrame([data], columns=FEATURE_COLS)
        scaled   = scaler.transform(df_input)
        pred     = model.predict(scaled)[0]
        proba    = model.predict_proba(scaled)[0]

        flood_prob    = round(float(proba[1]) * 100, 1)
        no_flood_prob = round(float(proba[0]) * 100, 1)
        is_flood      = bool(pred == 1)

        return render_template(
            "result.html",
            is_flood=is_flood,
            flood_prob=flood_prob,
            no_flood_prob=no_flood_prob,
            input_data=data,
        )

    except RuntimeError as e:
        flash(str(e), "danger")
        return redirect(url_for("predict"))
    except Exception as e:
        flash(f"Prediction error: {str(e)}", "danger")
        return redirect(url_for("predict"))


@app.route("/about")
def about():
    return render_template("about.html")


# ─── Run ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    port  = int(os.environ.get("PORT", 5000))
    debug = os.environ.get("FLASK_DEBUG", "True") == "True"
    app.run(host="0.0.0.0", port=port, debug=debug)
