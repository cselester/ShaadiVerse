💒 ShaadiVerse — AI Wedding Planner + Budget Optimizer

ShaadiVerse is an AI-powered wedding planning dashboard built using Gradio, LangChain, and Groq API, designed to simplify Indian wedding planning.
It helps you create personalized wedding schedules and optimize your budget with real vendor recommendations — all in one dashboard.

🌟 Features

🤖 AI Wedding Planner (Groq LLM)

Generates a day-by-day wedding schedule based on religion, location, and ceremonies.

Suggests realistic timelines and sequences (Haldi → Mehendi → Sangeet → Wedding → Reception).

💰 Smart Budget Optimizer

Auto-calculates budget allocations for Venue, Catering, Photography, Music Band, Decor, and Clothing.

Suggests vendors based on your location (currently supports Lucknow dataset).

📊 Interactive Dashboard

Clean, user-friendly interface built with Gradio.

Automatic data loading between Planner and Budget tabs.

Visual pie charts for budget allocations.

💾 Session Management

Automatically saves inputs to wedding_session.json and budget_session.json for seamless multi-agent flow.

🚀 How to Run Locally
1️⃣ Clone this repository
git clone https://github.com/<your-username>/shaadiverse-ai.git
cd shaadiverse-ai

2️⃣ Install dependencies
pip install -r requirements.txt

3️⃣ Add your Groq API key

Set your API key as an environment variable:

export GROQ_API_KEY="your_api_key_here"


(If you’re using Google Colab, store it with userdata.set('GROQ_API_KEY', 'your_api_key_here').)

4️⃣ Run the app
python app.py


The app will start on a local Gradio interface, usually at:

http://127.0.0.1:7860

🧩 Files Overview
File	Description
app.py	Main application file combining the Planner and Budget agents
requirements.txt	Python dependencies
wedding_session.json	Stores planner input data
budget_session.json	Stores budget recommendations
Venue - Sheet1.csv, Caterers - Sheet1.csv, etc.	Vendor datasets used for Lucknow recommendations
☁️ Deploy on Hugging Face Spaces

Create a new Space: https://huggingface.co/new-space

Choose Gradio as the SDK.

Upload:

app.py

requirements.txt

README.md

All vendor CSV files

Hugging Face will automatically install dependencies and launch your app.

🧠 Tech Stack

Frontend: Gradio (Tabs-based dashboard)

Backend: LangChain + Groq LLM (llama-3.3-70b-versatile)

Data: Pandas, NumPy

Visualization: Matplotlib

PDF (optional): fpdf2

Deployment: Hugging Face Spaces / Google Colab

👩‍💻 Example Use Case

Enter couple name, religion, and location.

Choose wedding dates and budget.

Click Generate Schedule → get a full AI-generated plan.

Switch to Budget Optimizer → instantly view category-wise allocations and vendor recommendations.

📬 Support & Contributions

Have ideas or feedback?
Feel free to open an Issue or Pull Request — contributions are welcome!
