# app.py — ShaadiVerse Wedding Planner + Budget Dashboard (Hugging Face Ready)
import os, json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime, timezone
import gradio as gr
from langchain_groq import ChatGroq

# -------------------------
# File paths
# -------------------------
VENUE_CSV = "Venue - Sheet1.csv"
CATERERS_CSV = "Caterers - Sheet1 (2).csv"
PHOTOGRAPHER_CSV = "Photographer - Sheet1 (1).csv"
MUSIC_CSV = "Music Band - Sheet1 (1).csv"
DECORATORS_CSV = "Decorators - Sheet1 (1).csv"
WEDDING_SESSION_FILE = "wedding_session.json"

# -------------------------
# Load data
# -------------------------
def safe_load_csv(path):
    if not os.path.exists(path):
        return pd.DataFrame()
    return pd.read_csv(path)

venues_df = safe_load_csv(VENUE_CSV)
caterers_df = safe_load_csv(CATERERS_CSV)
photographers_df = safe_load_csv(PHOTOGRAPHER_CSV)
music_df = safe_load_csv(MUSIC_CSV)
decorators_df = safe_load_csv(DECORATORS_CSV)

# -------------------------
# LLM Setup (Groq)
# -------------------------
groq_api_key = os.getenv("GROQ_API_KEY", "")
llm = ChatGroq(temperature=0.4, groq_api_key=groq_api_key, model_name="llama-3.3-70b-versatile")

# -------------------------
# Session handling
# -------------------------
def save_wedding_session(data):
    data_copy = data.copy()
    data_copy["saved_at"] = datetime.now(timezone.utc).isoformat()
    with open(WEDDING_SESSION_FILE, "w") as f:
        json.dump(data_copy, f, indent=2)

def load_wedding_session():
    if not os.path.exists(WEDDING_SESSION_FILE):
        return {}
    with open(WEDDING_SESSION_FILE, "r") as f:
        return json.load(f)

# -------------------------
# Planner
# -------------------------
def plan_with_groq(name, religion, start_date, end_date, budget, location, ceremonies, guests, clothing):
    session = {
        "name": name, "religion": religion, "start_date": start_date, "end_date": end_date,
        "budget": budget, "location": location, "ceremonies": ceremonies,
        "guests": guests, "clothing": clothing
    }
    save_wedding_session(session)
    prompt = f"""
    You are an AI Wedding Planner for ShaadiVerse.
    Couple Name: {name}
    Religion: {religion}
    Wedding Dates: {start_date} to {end_date}
    Ceremonies: {ceremonies}
    Task:
    - Create a complete wedding event schedule.
    - Include date, time, and brief notes for each event.
    - Ensure realistic gaps between events if us if user input.
    - Optimize sequence (e.g., Haldi → Mehendi → Sangeet → Wedding → Reception).
    - Follow Indian wedding customs for the selected religion.
    -
    Output format:
    1. [Ceremony Name] – [Date] at [Time]
       Notes: [Short description less than 30 words]
    """
    response = llm.invoke(prompt)
    text = response.content.strip()
    session["schedule_text"] = text
    save_wedding_session(session)
    return text

def update_schedule(existing_schedule, update_request):
    prompt = f"""Here is the current wedding schedule:
    {existing_schedule}
    User request: {update_request}
    Update the schedule accordingly.
    Maintain the same format and ensure timing/logical flow makes sense."""
    return llm.invoke(prompt).content.strip()

# -------------------------
# Budget Logic
# -------------------------
def allocate_budget(total_budget, guests, num_days, include_clothing=True):
    total = float(total_budget)
    base = {
        "Venue": 0.30, "Catering": 0.35, "Photographer": 0.10,
        "Music Band": 0.08, "Decorator": 0.07, "Clothing": 0.10
    }
    if not include_clothing:
        c_share = base.pop("Clothing")
        base["Catering"] += c_share * 0.5
        rem = c_share * 0.5
        for k in ["Venue", "Photographer", "Music Band", "Decorator"]:
            base[k] += rem * (base[k] / sum(base.values()))
    alloc = {k: round(v * total) for k, v in base.items()}
    diff = int(total) - sum(alloc.values())
    if diff:
        alloc["Catering"] += diff
    num_days = max(1, num_days)
    return {
        "allocations": alloc,
        "venue_per_day_target": round(alloc["Venue"] / num_days),
        "catering_per_plate_target": round(alloc["Catering"] / (guests + 100))
    }

def nearest_by_price(df, price_col, target, top_n=3):
    if df.empty or price_col not in df.columns:
        return pd.DataFrame([{"Note": "No data found"}])
    df2 = df.copy()
    df2[price_col] = pd.to_numeric(df2[price_col], errors="coerce").fillna(np.inf)
    df2["diff"] = (df2[price_col] - target).abs()
    return df2.sort_values("diff").head(top_n).drop(columns=["diff"])

def recommend_vendors(alloc_meta, num_days, guests, location):
    if not location or location.strip().lower() != "lucknow":
        msg = pd.DataFrame([{"Note": "Vendor data available only for Lucknow."}])
        return {"Venues": msg, "Caterers": msg, "Photographers": msg, "Music Bands": msg, "Decorators": msg}

    return {
        "Venues": nearest_by_price(venues_df, "Price", alloc_meta["venue_per_day_target"], 3),
        "Caterers": nearest_by_price(caterers_df, "Price", alloc_meta["catering_per_plate_target"], 5),
        "Photographers": nearest_by_price(photographers_df, "Price", alloc_meta["allocations"]["Photographer"], 3),
        "Music Bands": nearest_by_price(music_df, "Price", alloc_meta["allocations"]["Music Band"], 3),
        "Decorators": nearest_by_price(decorators_df, "Price", alloc_meta["allocations"]["Decorator"], 3)
    }

def plot_allocation_pie(alloc_meta):
    labels = list(alloc_meta["allocations"].keys())
    sizes = list(alloc_meta["allocations"].values())
    fig, ax = plt.subplots(figsize=(6,6))
    ax.pie(sizes, labels=labels, autopct=lambda p: f"₹{int(p/100*sum(sizes))}", startangle=140)
    ax.axis("equal")
    return fig

def run_full(name, religion, start_date, end_date, budget, location, ceremonies, guests, clothing):
    schedule = plan_with_groq(name, religion, start_date, end_date, budget, location, ceremonies, guests, clothing)
    try:
        sd = datetime.strptime(start_date, "%Y-%m-%d")
        ed = datetime.strptime(end_date, "%Y-%m-%d")
        num_days = (ed - sd).days + 1
    except:
        num_days = 1
    alloc_meta = allocate_budget(budget, guests, num_days, clothing)
    rec = recommend_vendors(alloc_meta, num_days, guests, location)
    alloc_df = pd.DataFrame(list(alloc_meta["allocations"].items()), columns=["Category", "Allocated (₹)"])
    fig = plot_allocation_pie(alloc_meta)
    return schedule, alloc_df, rec["Venues"], rec["Caterers"], rec["Photographers"], rec["Music Bands"], rec["Decorators"], fig

# -------------------------
# Gradio App
# -------------------------
sess = load_wedding_session()

with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.Markdown("# 💒 ShaadiVerse — Wedding Planner + Budget Optimizer")

    with gr.Tabs():
        with gr.TabItem("📅 Planner"):
            with gr.Row():
                name = gr.Textbox(label="Couple Name", value=sess.get("name",""))
                religion = gr.Dropdown(["Hindu","Sikh","Muslim","Christian","Other"], value=sess.get("religion","Hindu"))
            with gr.Row():
                start_date = gr.Textbox(label="Start Date (YYYY-MM-DD)", value=sess.get("start_date",""))
                end_date = gr.Textbox(label="End Date (YYYY-MM-DD)", value=sess.get("end_date",""))
            with gr.Row():
                budget = gr.Number(label="Total Budget (INR)", value=sess.get("budget",1500000))
                guests = gr.Number(label="Guests", value=sess.get("guests",300))
                clothing = gr.Checkbox(label="Include Clothing?", value=sess.get("clothing",True))
            location = gr.Textbox(label="Location", value=sess.get("location","Lucknow"))
            ceremonies = gr.Textbox(label="Ceremonies", value=sess.get("ceremonies","Haldi, Mehendi, Sangeet, Wedding, Reception"))
            gen_btn = gr.Button("Generate Schedule (AI)")
            schedule_out = gr.Textbox(label="Generated Schedule", lines=12)
            update_req = gr.Textbox(label="Update Request")
            update_btn = gr.Button("Update Schedule")
            gen_btn.click(plan_with_groq, [name,religion,start_date,end_date,budget,location,ceremonies,guests,clothing], schedule_out)
            update_btn.click(update_schedule, [schedule_out, update_req], schedule_out)

        with gr.TabItem("💰 Budget Optimizer"):
            run_btn = gr.Button("Run Full Planner + Budget")
            alloc_table = gr.Dataframe(label="Budget Allocation")
            venue_table = gr.Dataframe(label="Venues")
            caterer_table = gr.Dataframe(label="Caterers")
            photog_table = gr.Dataframe(label="Photographers")
            music_table = gr.Dataframe(label="Music Bands")
            decor_table = gr.Dataframe(label="Decorators")
            chart = gr.Plot(label="Budget Distribution")
            run_btn.click(run_full, [name,religion,start_date,end_date,budget,location,ceremonies,guests,clothing],
                          [schedule_out,alloc_table,venue_table,caterer_table,photog_table,music_table,decor_table,chart])

demo.launch()

