🧠 AI Sentiment Analysis Dashboard
Real-Time Market Mood Tracking • Machine Learning Forecasting • Modern Flask UI

📌 Project Overview

AI Sentiment Analysis Dashboard is an end-to-end Flask web application that processes real-time news & social media data, performs sentiment analysis, forecasts future trends using machine-learning algorithms, and provides a clean, modern interface to manage everything.

This project integrates:

Data Engineering

AI Sentiment Analysis

Forecasting (Prophet Model)

Flask Web Development

Slack Notifications

Interactive Dashboard Design

All wrapped into a polished, elegant dashboard.

⭐ Features
🔷 1. Pipeline Control Center

Control the full ingestion → analysis → forecast workflow with one click:

Fetch latest NewsAPI & Reddit posts

Run sentiment analysis

Update progress bar

View real-time logs

📊 2. Sentiment Analytics Visualization

Displays:

Positive, Neutral, Negative sentiment counts

Total records

Pie chart or bar chart

Clean white/blue UI

🔮 3. Sentiment Forecasting (Prophet)

Predict future sentiment trends using:

Daily averaging of sentiment

Facebook Prophet time-series model

Interactive forecast plot

Manual Slack alert trigger

🔔 4. Manual Slack Alerts

Alerts are only sent when you press the Alerts button, including:

Negative trend warnings

Forecast notifications

(Automatically sending alerts has been disabled as per project requirement.)

📑 5. Data Viewer

A complete viewer for:

Cleaned data

Sentiment-tagged records

Easy navigation and readability

⚙️ 6. Configuration Panel

Modify:

API keys

Queries

Subreddits

Slack webhook

Without touching backend code

🧱 Project Structure
MileStone-4/
│
├── app.py                     # Flask backend (routes & server)
├── milestone_2.py             # Data ingestion + sentiment analysis
├── milestone_3.py             # Forecasting + Prophet model
│
├── templates/
│   ├── index.html             # Main dashboard UI
│   └── data_view.html         # Data viewer UI
│
├── data/
│   └── analyzed_ai_market_data.csv
│
├── plots/
│   ├── sentiment_pie_chart.png
│   └── prophet_sentiment_forecast.png
│
├── .env                       # API keys (ignored in Git)
└── requirements.txt

🛠️ Technologies Used
Backend

Flask

Python

Pandas

Requests

Machine Learning

Facebook Prophet

CmdStanPy

Sentiment scoring (custom / Gemini / rule-based)

Frontend

HTML5

Bootstrap 5

FontAwesome

Modern glass-white UI theme

APIs

NewsAPI

Reddit API

Slack Webhooks

(Optional) Gemini API

📸 Screenshots

(Upload images and replace URLs below)

Dashboard

Forecast Plot

Data Viewer

🔧 Installation & Setup
1️⃣ Clone the repository
git clone https://github.com/YOUR-USERNAME/Milestone-4.git
cd Milestone-4

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Create .env file
NEWSAPI_KEY=xxxx
REDDIT_CLIENT_ID=xxxx
REDDIT_CLIENT_SECRET=xxxx
SLACK_WEBHOOK=xxxx
GEMINI_API_KEY=xxxx

4️⃣ Run the application
python app.py


Navigate to:
👉 http://127.0.0.1:5000

📡 API Keys Needed
API	Purpose
NewsAPI	Real-time news ingestion
Reddit API	Social sentiment collection
Slack Webhook	Alerts
Gemini AI (optional)	Sentiment interpretation
🧠 How It Works
Pipeline

Fetch news & Reddit posts

Clean & standardize text

Score sentiment

Save CSV

Update dashboard

Forecasting

Compute daily averages

Apply Prophet model

Generate future prediction

Save PNG

(Optional) User-triggered Slack alert

➕ Milestone-Specific Notes
✔ milestone_2.py

Handles ingestion, cleaning, and sentiment scoring

Does NOT trigger Slack automatically

Fully controlled via dashboard

✔ milestone_3.py

Builds the Prophet forecasting model

Only sends Slack alerts when user clicks Send Alert button

No auto-alerts

👨‍💻 For Developers / Contributors
Run linters
flake8 .

Format code
black .

Create pull request

Fork repo

Create feature branch

Push your changes

Submit PR

🪪 License

This project is licensed under the MIT License.

⭐ Star the Repository!

If you found this project useful or inspiring, please consider giving it a ⭐ on GitHub!
