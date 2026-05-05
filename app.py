import os
import time
import io
import inspect
import logging
import warnings
import contextlib
import datetime as dt
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import yfinance as yf
from sklearn.preprocessing import MinMaxScaler
from flask import Flask, render_template, request, send_file
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import InputLayer
from dotenv import load_dotenv

# =================================================
# BASIC APP SETUP
# =================================================
load_dotenv()

STATIC_DIR = os.getenv("STATIC_DIR", "static")
MODEL_PATH = os.getenv("MODEL_PATH", "stock_price_prediction.keras")
DEFAULT_STOCK = os.getenv("DEFAULT_STOCK", "RELIANCE.NS")
PREDICTION_WINDOW = int(os.getenv("PREDICTION_WINDOW", "100"))
START_DATE = os.getenv("START_DATE", "2010-01-01")

app = Flask(__name__)
app.config["SEND_FILE_MAX_AGE_DEFAULT"] = 0

model = None

# Reduce console noise from yfinance / urllib3
warnings.filterwarnings("ignore")
logging.getLogger("yfinance").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)
logging.getLogger("matplotlib").setLevel(logging.ERROR)

# =================================================
# KERAS / INPUTLAYER PATCH
# Helps with old/new Keras config mismatches
# =================================================
_original_inputlayer_init = InputLayer.__init__
_original_inputlayer_sig = inspect.signature(_original_inputlayer_init)

def _patched_inputlayer_init(self, *args, **kwargs):
    kwargs.pop("optional", None)

    batch_shape = kwargs.pop("batch_shape", None)
    batch_input_shape = kwargs.pop("batch_input_shape", None)

    if batch_shape is None and batch_input_shape is not None:
        batch_shape = batch_input_shape

    if batch_shape is not None:
        batch_shape = tuple(batch_shape)

        if "batch_input_shape" in _original_inputlayer_sig.parameters:
            kwargs.setdefault("batch_input_shape", batch_shape)
        elif "batch_shape" in _original_inputlayer_sig.parameters:
            kwargs.setdefault("batch_shape", batch_shape)
        elif "shape" in _original_inputlayer_sig.parameters:
            if len(batch_shape) >= 2:
                kwargs.setdefault("shape", tuple(batch_shape[1:]))
            else:
                kwargs.setdefault("shape", batch_shape)

    return _original_inputlayer_init(self, *args, **kwargs)

InputLayer.__init__ = _patched_inputlayer_init

# =================================================
# MODEL LOADING
# =================================================
def load_stock_model():
    global model

    if not os.path.exists(MODEL_PATH):
        print(f"⚠ Model file not found: {MODEL_PATH}")
        model = None
        return

    load_attempts = [
        {"compile": False, "safe_mode": False},
        {"compile": False},
        {"compile": True, "safe_mode": False},
        {"compile": True},
    ]

    for kwargs in load_attempts:
        try:
            # Some Keras versions accept safe_mode, some do not
            try:
                sig = inspect.signature(load_model)
                if "safe_mode" not in sig.parameters:
                    kwargs = {k: v for k, v in kwargs.items() if k != "safe_mode"}
            except Exception:
                kwargs = {k: v for k, v in kwargs.items() if k != "safe_mode"}

            with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
                model = load_model(MODEL_PATH, **kwargs)

            print("✅ Model loaded successfully")
            return

        except Exception as e:
            print(f"⚠ Model load attempt failed with {kwargs}: {e}")

    print("❌ Model could not be loaded. App will run with fallback prediction only.")
    model = None

load_stock_model()

# =================================================
# YFINANCE HELPERS
# =================================================
def _silence_output(fn, *args, **kwargs):
    """Run noisy functions quietly."""
    buf_out = io.StringIO()
    buf_err = io.StringIO()
    with contextlib.redirect_stdout(buf_out), contextlib.redirect_stderr(buf_err):
        return fn(*args, **kwargs)

def normalize_stock_symbol(symbol: str) -> str:
    symbol = (symbol or "").strip().upper()
    symbol = symbol.replace(" ", "")
    return symbol

def build_symbol_candidates(stock: str):
    """
    Creates a small, sensible fallback list for Indian tickers.
    Example:
        RELIANCE.NS -> RELIANCE.NS, RELIANCE.BO, RELIANCE
        RELIANCE    -> RELIANCE, RELIANCE.NS, RELIANCE.BO
    """
    stock = normalize_stock_symbol(stock)
    if not stock:
        return []

    candidates = [stock]

    if "." in stock:
        base, suffix = stock.rsplit(".", 1)
        if suffix == "NS":
            candidates.extend([f"{base}.BO", base])
        elif suffix == "BO":
            candidates.extend([f"{base}.NS", base])
        else:
            candidates.extend([base, f"{base}.NS", f"{base}.BO"])
    else:
        candidates.extend([f"{stock}.NS", f"{stock}.BO"])

    # De-duplicate while preserving order
    seen = set()
    out = []
    for x in candidates:
        if x and x not in seen:
            seen.add(x)
            out.append(x)
    return out

def fetch_yf_once(symbol: str, start: str = START_DATE):
    """
    Fetch using yfinance quietly.
    Tries Ticker.history first, then download as a backup.
    """
    end = dt.datetime.now()

    # Attempt 1: Ticker.history
    try:
        ticker = yf.Ticker(symbol)
        df = _silence_output(
            ticker.history,
            start=start,
            end=end,
            auto_adjust=True,
            actions=False,
            interval="1d"
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    # Attempt 2: yf.download backup
    try:
        df = _silence_output(
            yf.download,
            symbol,
            start=start,
            end=end,
            progress=False,
            threads=False,
            auto_adjust=True,
            group_by="column"
        )
        if isinstance(df, pd.DataFrame) and not df.empty:
            return df
    except Exception:
        pass

    return None

def fetch_stock_data(stock: str, retries: int = 2, sleep_seconds: float = 0.75):
    """
    Tries multiple symbol variants quietly until one returns data.
    Returns: (df, resolved_symbol)
    """
    stock = normalize_stock_symbol(stock) or DEFAULT_STOCK
    candidates = build_symbol_candidates(stock)

    # For common Indian names, add a few helpful extra options if the user enters a bare name.
    extra_map = {
        "RELIANCE": ["RELIANCE.NS", "RELIANCE.BO"],
        "TCS": ["TCS.NS", "TCS.BO"],
        "INFY": ["INFY.NS", "INFY.BO"],
        "SBIN": ["SBIN.NS", "SBIN.BO"],
        "HDFCBANK": ["HDFCBANK.NS", "HDFCBANK.BO"],
        "ICICIBANK": ["ICICIBANK.NS", "ICICIBANK.BO"],
    }
    if stock in extra_map:
        for s in extra_map[stock]:
            if s not in candidates:
                candidates.append(s)

    last_error = None

    for candidate in candidates:
        for attempt in range(1, retries + 1):
            try:
                df = fetch_yf_once(candidate)
                if df is not None and not df.empty:
                    return df, candidate
            except Exception as e:
                last_error = e

            time.sleep(sleep_seconds)

    if last_error is not None:
        print(f"⚠ Final data fetch error for {stock}: {last_error}")

    return None, stock

# =================================================
# DATAFRAME / SERIES HELPERS
# =================================================
def normalize_dataframe_columns(df: pd.DataFrame) -> pd.DataFrame:
    if df is None or df.empty:
        return df

    df = df.copy()

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [
            "_".join([str(x) for x in col if str(x) != ""]).strip()
            for col in df.columns.values
        ]

    return df

def extract_close_series(df: pd.DataFrame):
    if df is None or df.empty:
        return None

    df = normalize_dataframe_columns(df)

    if "Close" not in df.columns:
        for col in df.columns:
            if str(col).lower() == "close":
                df["Close"] = df[col]
                break

    if "Close" not in df.columns:
        return None

    close_raw = df["Close"]
    if isinstance(close_raw, pd.DataFrame):
        close_raw = close_raw.iloc[:, 0]

    close_series = pd.to_numeric(close_raw, errors="coerce").dropna()
    if close_series.empty:
        return None

    return close_series.astype(float)

def create_sequences(data: np.ndarray, window: int = 100):
    x, y = [], []
    for i in range(window, len(data)):
        x.append(data[i - window:i])
        y.append(data[i, 0])
    return np.array(x), np.array(y)

def safe_inverse_scale(values, scaler: MinMaxScaler):
    """
    Returns inverse-transformed values only when they look scaled.
    If they already look like raw prices, returns them unchanged.
    """
    arr = np.asarray(values, dtype=float).reshape(-1)

    if arr.size == 0:
        return arr

    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return arr

    looks_scaled = finite.min() >= -0.25 and finite.max() <= 1.25

    if looks_scaled:
        try:
            return scaler.inverse_transform(arr.reshape(-1, 1)).reshape(-1)
        except Exception:
            return arr

    return arr

def last_n_trend(close_series: pd.Series, n: int = 5):
    close = close_series.dropna()
    if len(close) < 2:
        return 0.0

    n = min(n, len(close) - 1)
    start_price = float(close.iloc[-(n + 1)])
    end_price = float(close.iloc[-1])

    if start_price == 0:
        return 0.0

    return (end_price - start_price) / start_price

def calculate_rsi(close_series: pd.Series, period: int = 14):
    close = close_series.dropna()
    if len(close) < period + 1:
        return None

    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window=period).mean()
    loss = (-delta.clip(upper=0)).rolling(window=period).mean()

    rs = gain / loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    if rsi.dropna().empty:
        return None
    return float(rsi.iloc[-1])

def calculate_macd(close_series: pd.Series):
    close = close_series.dropna()
    if len(close) < 35:
        return None, None, None

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd = ema12 - ema26
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return float(macd.iloc[-1]), float(signal.iloc[-1]), float(hist.iloc[-1])

# =================================================
# PLOT HELPERS
# =================================================
def save_plot(path, plot_fn):
    plt.figure(figsize=(11, 5))
    plot_fn()
    plt.legend()
    plt.tight_layout()
    plt.savefig(path, dpi=150)
    plt.close()

def ensure_static_dir():
    os.makedirs(STATIC_DIR, exist_ok=True)

# =================================================
# RECOMMENDATION LOGIC
# =================================================
def generate_recommendation(
    close_series: pd.Series,
    predicted_price: float = None,
    model_used: bool = False,
    ema20_last: float = None,
    ema50_last: float = None
):
    close = close_series.dropna().astype(float)

    if close.empty:
        return {
            "action": "HOLD ⏳",
            "confidence": "Low",
            "reason": "Not enough market data is available to make a reliable recommendation.",
            "last_close": None,
            "predicted_price": None,
            "predicted_move_pct": None,
            "trend_pct": None,
        }

    last_close = float(close.iloc[-1])
    recent_trend = float(last_n_trend(close, n=5))

    if predicted_price is None or not np.isfinite(predicted_price):
        predicted_price = last_close * (1 + recent_trend)

    predicted_price = float(predicted_price)

    if last_close == 0:
        predicted_move = 0.0
    else:
        predicted_move = (predicted_price - last_close) / last_close

    ema_bonus = 0.0
    ema_comment = ""

    if ema20_last is not None and ema50_last is not None:
        if ema20_last > ema50_last:
            ema_bonus = 0.004
            ema_comment = "Short-term momentum is above the medium-term trend."
        elif ema20_last < ema50_last:
            ema_bonus = -0.004
            ema_comment = "Short-term momentum is below the medium-term trend."
        else:
            ema_comment = "Short-term and medium-term momentum are balanced."

    rsi = calculate_rsi(close)
    rsi_bonus = 0.0
    rsi_comment = ""

    if rsi is not None:
        if rsi < 30:
            rsi_bonus = 0.003
            rsi_comment = "RSI suggests the stock may be oversold."
        elif rsi > 70:
            rsi_bonus = -0.003
            rsi_comment = "RSI suggests the stock may be overbought."
        else:
            rsi_comment = "RSI is in a neutral zone."

    score = (0.65 * predicted_move) + (0.25 * recent_trend) + ema_bonus + rsi_bonus

    if score > 0.01:
        action = "BUY 📈"
        confidence = "High" if score > 0.03 else "Medium" if score > 0.015 else "Low"
        base_reason = "The stock shows signs of upward momentum."
    elif score < -0.01:
        action = "SELL 📉"
        confidence = "High" if score < -0.03 else "Medium" if score < -0.015 else "Low"
        base_reason = "The stock shows signs of downward pressure."
    else:
        action = "HOLD ⏳"
        confidence = "Low"
        base_reason = "The stock is moving without a strong directional edge."

    trend_text = "upward" if recent_trend > 0 else "downward" if recent_trend < 0 else "sideways"
    model_text = "The model helped shape this view." if model_used else "This view is based on recent market behavior."

    details = [base_reason, f"Recent trend looks {trend_text}."]
    if ema_comment:
        details.append(ema_comment)
    if rsi_comment:
        details.append(rsi_comment)
    details.append(model_text)

    human_reason = " ".join(details)

    return {
        "action": action,
        "confidence": confidence,
        "reason": human_reason,
        "last_close": round(last_close, 2),
        "predicted_price": round(predicted_price, 2),
        "predicted_move_pct": round(predicted_move * 100, 2),
        "trend_pct": round(recent_trend * 100, 2),
        "rsi": None if rsi is None else round(rsi, 2),
    }

# =================================================
# MAIN PROCESSING
# =================================================
def process_prediction(stock_input, template="prediction.html"):
    stock_input = normalize_stock_symbol(stock_input) or DEFAULT_STOCK

    df, resolved_stock = fetch_stock_data(stock_input)

    if df is None or df.empty:
        return render_template(
            template,
            error_message="Stock data unavailable. Please try another ticker.",
            stock_value=stock_input,
            resolved_stock=resolved_stock,
            recommendation=None,
            data_desc=None,
            dataset_link=None,
            plot_path_ema_20_50=None,
            plot_path_ema_100_200=None,
            plot_path_prediction=None
        )

    df = normalize_dataframe_columns(df).dropna(how="all")

    close_series = extract_close_series(df)
    if close_series is None or len(close_series) < max(PREDICTION_WINDOW + 50, 150):
        return render_template(
            template,
            error_message="Not enough valid price data for prediction.",
            stock_value=stock_input,
            resolved_stock=resolved_stock,
            recommendation=None,
            data_desc=None,
            dataset_link=None,
            plot_path_ema_20_50=None,
            plot_path_ema_100_200=None,
            plot_path_prediction=None
        )

    # Keep a compact, readable description table for the UI
    try:
        data_desc = df.describe(include="all").fillna("").to_html(classes="table-auto w-full", border=0)
    except Exception:
        data_desc = None

    # Technical indicators
    ema20 = close_series.ewm(span=20, adjust=False).mean()
    ema50 = close_series.ewm(span=50, adjust=False).mean()
    ema100 = close_series.ewm(span=100, adjust=False).mean()
    ema200 = close_series.ewm(span=200, adjust=False).mean()

    # Train / test split
    close_df = pd.DataFrame(close_series.values, columns=["Close"])
    train_size = int(len(close_df) * 0.7)

    if train_size <= PREDICTION_WINDOW:
        train_size = PREDICTION_WINDOW + 1

    train = close_df.iloc[:train_size].copy()
    test = close_df.iloc[train_size:].copy()

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaler.fit(train)

    # Build the final input block for sequence generation
    past_window = train.tail(PREDICTION_WINDOW)
    final_df = pd.concat([past_window, test], ignore_index=True)

    input_data = scaler.transform(final_df)
    x_test, y_test = create_sequences(input_data, window=PREDICTION_WINDOW)

    y_predicted = None
    y_test_actual = None
    predicted_price = None
    model_used = False

    if model is not None and len(x_test) > 0:
        try:
            x_test_reshaped = x_test.reshape((x_test.shape[0], x_test.shape[1], 1))

            pred_scaled = model.predict(x_test_reshaped, verbose=0)
            y_predicted = safe_inverse_scale(pred_scaled, scaler)
            y_test_actual = safe_inverse_scale(y_test, scaler)

            if y_predicted is not None and len(y_predicted) > 0:
                predicted_price = float(np.asarray(y_predicted).reshape(-1)[-1])
                model_used = True

        except Exception as e:
            print(f"❌ Prediction error: {e}")
            model_used = False
            predicted_price = None

    # Fallback prediction when model fails or is unavailable
    if predicted_price is None:
        recent_trend = last_n_trend(close_series, n=5)
        ema_trend = 0.0
        if len(ema20.dropna()) > 0 and len(ema50.dropna()) > 0:
            ema_trend = (float(ema20.iloc[-1]) - float(ema50.iloc[-1])) / float(close_series.iloc[-1])

        blended_trend = (0.7 * recent_trend) + (0.3 * ema_trend)
        predicted_price = float(close_series.iloc[-1]) * (1 + blended_trend)

    recommendation = generate_recommendation(
        close_series=close_series,
        predicted_price=predicted_price,
        model_used=model_used,
        ema20_last=float(ema20.iloc[-1]) if len(ema20) else None,
        ema50_last=float(ema50.iloc[-1]) if len(ema50) else None,
    )

    ensure_static_dir()

    # Plots
    ema_20_50_path = os.path.join(STATIC_DIR, "ema_20_50.png")
    save_plot(
        ema_20_50_path,
        lambda: (
            plt.plot(close_series.values, label="Close"),
            plt.plot(ema20.values, label="EMA 20"),
            plt.plot(ema50.values, label="EMA 50"),
            plt.title(f"{resolved_stock} - Close vs EMA 20/50"),
            plt.xlabel("Time"),
            plt.ylabel("Price"),
        )
    )

    ema_100_200_path = os.path.join(STATIC_DIR, "ema_100_200.png")
    save_plot(
        ema_100_200_path,
        lambda: (
            plt.plot(close_series.values, label="Close"),
            plt.plot(ema100.values, label="EMA 100"),
            plt.plot(ema200.values, label="EMA 200"),
            plt.title(f"{resolved_stock} - Close vs EMA 100/200"),
            plt.xlabel("Time"),
            plt.ylabel("Price"),
        )
    )

    prediction_path = None
    if y_predicted is not None and y_test_actual is not None and len(y_predicted) > 0 and len(y_test_actual) > 0:
        prediction_path = os.path.join(STATIC_DIR, "prediction.png")
        save_plot(
            prediction_path,
            lambda: (
                plt.plot(np.asarray(y_test_actual).reshape(-1), label="Actual"),
                plt.plot(np.asarray(y_predicted).reshape(-1), label="Predicted"),
                plt.title(f"{resolved_stock} - Actual vs Predicted"),
                plt.xlabel("Time"),
                plt.ylabel("Price"),
            )
        )

    # Save CSV for download
    csv_name = f"{resolved_stock.replace('.', '_')}.csv"
    csv_path = os.path.join(STATIC_DIR, csv_name)

    try:
        df.to_csv(csv_path)
    except Exception as e:
        print(f"⚠ Could not save CSV: {e}")
        csv_name = None

    return render_template(
        template,
        error_message=None,
        stock_value=stock_input,
        resolved_stock=resolved_stock,
        recommendation=recommendation,
        data_desc=data_desc,
        dataset_link=csv_name,
        plot_path_ema_20_50=ema_20_50_path,
        plot_path_ema_100_200=ema_100_200_path,
        plot_path_prediction=prediction_path,
    )

# =================================================
# ROUTES
# =================================================
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        return process_prediction(request.form.get("stock"))
    return render_template("home.html")

@app.route("/prediction", methods=["GET", "POST"])
def prediction():
    if request.method == "POST":
        return process_prediction(request.form.get("stock"))
    return render_template(
        "prediction.html",
        stock_value=DEFAULT_STOCK,
        resolved_stock=DEFAULT_STOCK,
        recommendation=None,
        error_message=None,
        data_desc=None,
        dataset_link=None,
        plot_path_ema_20_50=None,
        plot_path_ema_100_200=None,
        plot_path_prediction=None
    )

@app.route("/download/<filename>")
def download_file(filename):
    path = os.path.join(STATIC_DIR, filename)
    if os.path.exists(path):
        return send_file(path, as_attachment=True)
    return "File not found", 404

@app.route("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": model is not None,
        "default_stock": DEFAULT_STOCK,
    }

# =================================================
# RUN APP
# =================================================
if __name__ == "__main__":
    host = os.getenv("FLASK_HOST", "127.0.0.1")
    port = int(os.getenv("FLASK_PORT", "5000"))
    debug_mode = os.getenv("FLASK_DEBUG", "1") == "1"

    app.run(host=host, port=port, debug=debug_mode, use_reloader=False)