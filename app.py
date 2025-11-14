from flask import Flask, render_template, jsonify, send_file
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
import io
import base64
import os
import threading
from datetime import datetime
from dotenv import load_dotenv
from prophet import Prophet
import matplotlib

matplotlib.use("Agg")  # Non-GUI backend

# ----------------------------
# Flask setup
# ----------------------------
app = Flask(__name__)
app.secret_key = "ai-sentiment-dashboard-2024"
load_dotenv()

# ----------------------------
# Imports
# ----------------------------
from milestone_2 import (
    fetch_newsapi_articles,
    fetch_reddit_posts,
    analyze_sentiments,
    AI_QUERIES,
    REDDIT_SUBREDDITS,
    send_slack_alert,
)
from milestone_3 import load_data, run_prophet_forecast

# ----------------------------
# Configuration
# ----------------------------
DATA_DIR = "/tmp/ai_sentiment_data"
os.makedirs(DATA_DIR, exist_ok=True)
sns.set_style("whitegrid")
plt.rcParams["figure.figsize"] = (12, 8)
plot_lock = threading.Lock()

# ----------------------------
# Pipeline Manager
# ----------------------------
class PipelineManager:
    def __init__(self):
        self.status = "stopped"
        self.last_run = None
        self.progress = 0
        self.logs = []

    def start_pipeline(self):
        if self.status == "running":
            return False
        self.status = "running"
        self.progress = 0
        self.logs = [f"🚀 Starting AI Sentiment Pipeline at {datetime.now().strftime('%H:%M:%S')}"]
        threading.Thread(target=self._run_real_pipeline, daemon=True).start()
        return True

    def _run_real_pipeline(self):
        try:
            self._update(20, "📰 Fetching news articles...")
            news = fetch_newsapi_articles(AI_QUERIES, total_records=30)

            self._update(40, "🔴 Fetching Reddit posts...")
            reddit = fetch_reddit_posts(REDDIT_SUBREDDITS, total_records=30)

            self._update(60, "🔄 Combining sources...")
            rows = news + reddit
            df = pd.DataFrame(rows)
            if "platform" not in df.columns:
                df["platform"] = "unknown"
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            df = df[["platform", "timestamp", "query", "text", "url"]]
            df.to_csv(f"{DATA_DIR}/raw_data.csv", index=False)

            self._update(80, "🤖 Analyzing sentiments...")
            analyzed = analyze_sentiments(df, batch_size=5)
            analyzed.to_csv(f"{DATA_DIR}/analyzed_data.csv", index=False)

            self._update(100, "✅ Pipeline complete!")
            self.status = "completed"
            self.last_run = datetime.now()
        except Exception as e:
            import traceback
            self.status = "error"
            self.logs.append(f"❌ ERROR: {e}")
            self.logs.append(traceback.format_exc())

    def _update(self, progress, msg):
        self.progress = progress
        self.logs.append(f"{datetime.now().strftime('%H:%M:%S')}: {msg}")


# ----------------------------
# Forecast Manager
# ----------------------------
class ForecastManager:
    def __init__(self):
        self.status = "stopped"
        self.last_run = None
        self.progress = 0
        self.logs = []
        self.forecast_plot = None

    def start_forecast(self):
        if self.status == "running":
            return False
        self.status = "running"
        self.progress = 0
        self.logs = [f"📈 Starting Prophet Forecast at {datetime.now().strftime('%H:%M:%S')}"]
        threading.Thread(target=self._run_forecast, daemon=True).start()
        return True

    def _run_forecast(self):
        try:
            self._update(30, "📊 Loading sentiment data...")
            df = load_data()
            forecast = run_prophet_forecast(df)

            self._update(80, "🧠 Generating visualization...")
            self.forecast_plot = self._make_plot(df, forecast)

            self._update(100, "✅ Forecast complete!")
            self.status = "completed"
            self.last_run = datetime.now()
        except Exception as e:
            import traceback
            self.status = "error"
            self.logs.append(f"❌ Forecast error: {e}")
            self.logs.append(traceback.format_exc())

    def _update(self, progress, msg):
        self.progress = progress
        self.logs.append(f"{datetime.now().strftime('%H:%M:%S')}: {msg}")

    def _make_plot(self, df, forecast):
        try:
            with plot_lock:
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10))
                ax1.plot(df["ds"], df["y"], "bo-", alpha=0.7, label="Historical")
                ax1.plot(forecast["ds"], forecast["yhat"], "r-", label="Forecast")
                ax1.fill_between(
                    forecast["ds"],
                    forecast["yhat_lower"],
                    forecast["yhat_upper"],
                    color="red",
                    alpha=0.2,
                    label="Uncertainty",
                )
                ax1.set_title("AI Market Sentiment Forecast", fontweight="bold")
                ax1.legend()
                ax1.grid(True, alpha=0.3)

                model = Prophet(daily_seasonality=True, weekly_seasonality=True)
                model.fit(df)
                ax2.plot(forecast["ds"], forecast["trend"], "g-", linewidth=2, label="Trend")
                ax2.legend()
                ax2.grid(True, alpha=0.3)

                plt.tight_layout()
                buf = io.BytesIO()
                plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
                buf.seek(0)
                img_b64 = base64.b64encode(buf.read()).decode()
                plt.close()
                return f"data:image/png;base64,{img_b64}"
        except Exception as e:
            print("Forecast plot error:", e)
            return None


# ----------------------------
# Helpers
# ----------------------------
pipeline_manager = PipelineManager()
forecast_manager = ForecastManager()


def get_data():
    path = f"{DATA_DIR}/analyzed_data.csv"
    if os.path.exists(path):
        try:
            df = pd.read_csv(path)
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
            return df
        except Exception as e:
            print("Data load error:", e)
    return pd.DataFrame()


def make_charts():
    df = get_data()
    if df.empty:
        return None
    try:
        with plot_lock:
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle("AI Market Sentiment Dashboard", fontsize=16, fontweight="bold")

            plt.sca(axes[0, 0])
            sns.countplot(data=df, x="label", hue="platform", palette="Set2")
            plt.title("Sentiment by Platform", fontweight="bold")

            plt.sca(axes[0, 1])
            counts = df["label"].value_counts()
            plt.pie(
                counts.values,
                labels=counts.index,
                autopct="%1.1f%%",
                colors=["#2ecc71", "#f39c12", "#e74c3c"],
                startangle=90,
            )
            plt.title("Overall Sentiment", fontweight="bold")

            plt.sca(axes[1, 0])
            if "score" in df.columns:
                sns.histplot(df, x="score", hue="label", multiple="stack", bins=20)
                plt.title("Score Distribution", fontweight="bold")

            plt.sca(axes[1, 1])
            text_data = df["text"].dropna().astype(str)
            if text_data.empty:
                plt.text(0.5, 0.5, "No Text Data", ha="center", va="center")
            else:
                wc = WordCloud(width=400, height=300, background_color="white").generate(
                    " ".join(text_data)
                )
                plt.imshow(wc, interpolation="bilinear")
                plt.axis("off")
            plt.title("Word Cloud", fontweight="bold")

            plt.tight_layout()
            buf = io.BytesIO()
            plt.savefig(buf, format="png", bbox_inches="tight", dpi=100)
            buf.seek(0)
            img_b64 = base64.b64encode(buf.read()).decode()
            plt.close()
            return f"data:image/png;base64,{img_b64}"
    except Exception as e:
        print("Chart error:", e)
        return None


# ----------------------------
# Routes
# ----------------------------
@app.route("/")
def home():
    df = get_data()
    total = len(df)
    pos = len(df[df["label"] == "positive"]) if total else 0
    neu = len(df[df["label"] == "neutral"]) if total else 0
    neg = len(df[df["label"] == "negative"]) if total else 0

    stats = {
        "total_records": total,
        "positive_count": pos,
        "neutral_count": neu,
        "negative_count": neg,
    }

    return render_template("index.html", stats=stats, chart_url=None, forecast_plot=None)


@app.route("/charts")
def charts():
    chart = make_charts()
    if not chart:
        return jsonify({"error": "No chart available"}), 404
    return jsonify({"chart": chart})


@app.route("/start_pipeline", methods=["POST"])
def start_pipeline():
    return jsonify({"status": "started" if pipeline_manager.start_pipeline() else "already_running"})


@app.route("/run_forecast", methods=["POST"])
def run_forecast():
    return jsonify({"status": "started" if forecast_manager.start_forecast() else "already_running"})


@app.route("/pipeline_status")
def pipeline_status():
    return jsonify(
        {
            "status": pipeline_manager.status,
            "progress": pipeline_manager.progress,
            "logs": pipeline_manager.logs[-20:],
            "last_run": pipeline_manager.last_run.isoformat() if pipeline_manager.last_run else None,
        }
    )


@app.route("/forecast_status")
def forecast_status():
    return jsonify(
        {
            "status": forecast_manager.status,
            "progress": forecast_manager.progress,
            "logs": forecast_manager.logs[-20:],
            "last_run": forecast_manager.last_run.isoformat() if forecast_manager.last_run else None,
        }
    )


@app.route("/export_data")
def export_data():
    df = get_data()
    if df.empty:
        return "No data available"
    buf = io.BytesIO()
    df.to_csv(buf, index=False)
    buf.seek(0)
    return send_file(
        buf,
        mimetype="text/csv",
        as_attachment=True,
        download_name=f"sentiment_data_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
    )


@app.route("/view_data")
def view_data():
    path = f"{DATA_DIR}/analyzed_data.csv"
    if not os.path.exists(path):
        return render_template("data_view.html", data=[])
    df = pd.read_csv(path)
    return render_template(
        "data_view.html", tables=[df.to_html(classes="table table-striped", index=False)]
    )


# ----------------------------
# Entry point
# ----------------------------
if __name__ == "__main__":
    print("🚀 Starting AI Sentiment Dashboard...")
    print("🌐 Visit: http://localhost:5000")
    app.run(debug=False, use_reloader=False, host="0.0.0.0", port=5000)
