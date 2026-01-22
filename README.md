💰 AI Expense Tracker (Powered by GPT-4o)

A full-stack personal finance application that uses Natural Language Processing (LLMs) to categorize expenses automatically.

(Replace this line with an actual screenshot of your dashboard)

🚀 The Problem

Tracking expenses is tedious. Most apps require you to manually select dates, categories, and input amounts in specific fields. I wanted a solution where I could just "talk" to my finance tracker.

💡 The Solution

I built a Natural Language Expense Tracker. Instead of forms, users simply type:

"Spent $45 on dinner at Mario's Italian"

The system uses OpenAI's GPT-4o-mini to intelligently parse this string into structured JSON:

{
  "description": "Dinner at Mario's Italian",
  "amount": 45.00,
  "category": "Food",
  "date": "2024-03-21"
}


🛠️ Tech Stack

Backend (Python)

FastAPI: For high-performance async API endpoints.

SQLAlchemy (Async): For non-blocking database interactions.

OpenAI API: Using gpt-4o-mini for the parsing engine.

Pydantic: For strict data validation and schema management.

OAuth2 + JWT: Secure authentication flow.

Frontend (React/TypeScript)

Next.js 14: Using the App Router for modern routing and server components.

Tailwind CSS: For a responsive, dark-mode UI.

Recharts: For data visualization (Pie Charts).

Axios Interceptors: For automatic token management.

✨ Key Features

🪄 Magic Parse: Type expenses in plain English. The AI handles the math and categorization.

📊 Visual Analytics: Interactive Pie Charts showing spending breakdown.

🎯 Budget Goals: Set monthly limits and see real-time progress bars.

🔒 Secure Auth: Complete User Registration & Login system with JWTs.

⚡ Smart Filtering: Filter history by month, category, or price range.

🏗️ How to Run Locally

Prerequisites

Node.js & npm

Python 3.10+

OpenAI API Key

1. Clone & Setup Backend

cd backend
python -m venv venv
# Windows: .\venv\Scripts\activate
# Mac/Linux: source venv/bin/activate
pip install -r requirements.txt

# Set up environment variables
# Create a .env file and add: 
# OPENAI_API_KEY="your_key_here"
# SECRET_KEY="your_secret_here"

# Run Server
python main.py


2. Setup Frontend

cd frontend
npm install
npm run dev


Visit http://localhost:3000 to start tracking!

🔮 Future Roadmap

Receipt Scanning: Integration with GPT-4-Vision to extract data from photos.

Voice Input: Using WebSpeech API for hands-free logging.

Export: CSV/Excel export for tax season.

👨‍💻 Author

Built by [Your Name] as an exploration into AI Engineering.
[LinkedIn Profile] | [GitHub Profile]
