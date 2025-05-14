import streamlit as st
import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
from textblob import TextBlob

# App title
st.set_page_config(page_title="Daily AI Journal", layout="centered")
st.title("📝 Daily Personalized AI Journal")
st.markdown("Write your journal and get a thoughtful reply from your AI coach, mentor, or friend.")

# Gemini API Key Input
api_key = st.text_input("Enter your Gemini API Key", type="password")
if not api_key:
    st.warning("Please enter your Gemini API key to continue.")
    st.stop()

# Configure Gemini API
genai.configure(api_key=api_key)
model = genai.GenerativeModel("gemini-2.5-flash-preview-04-17")

# Prompt builder function
def build_prompt(entry, mode):
    mode_instructions = {
        "Coach": "Reply like a motivational coach. Give positive reinforcement, practical tips, and encourage action.",
        "Mentor": "Reply like a wise mentor. Provide insightful reflections, thought-provoking questions, and guidance.",
        "Friend": "Reply like a supportive best friend. Be kind, empathetic, and emotionally engaging.",
    }
    return f"""You are a {mode.lower()} AI. A user has shared their personal journal entry for today. Your job is to respond supportively and reflectively in a tone matching the role of a {mode.lower()}.

    Journal Entry:
    {entry}

    Your Response:"""

# Mode selection
mode = st.selectbox("Choose AI Role:", ["Coach", "Mentor", "Friend"])

# Input journal
entry = st.text_area("Your Journal Entry", height=250)

if "history" not in st.session_state:
    st.session_state.history = []

# Submit
if st.button("Reflect with AI"):
    with st.spinner("Thinking..."):
        prompt = build_prompt(entry, mode)
        response = model.generate_content(prompt)
        sentiment = TextBlob(entry).sentiment.polarity
        mood = "😊 Positive" if sentiment > 0.2 else "😐 Neutral" if sentiment >= -0.2 else "😟 Negative"

        # Store session
        st.session_state.history.append({
            "date": datetime.date.today().isoformat(),
            "entry": entry,
            "response": response.text,
            "sentiment": sentiment
        })

        # Display results
        st.markdown("### 🤖 AI's Response")
        st.write(response.text)

        st.markdown(f"### 🧠 Sentiment: {mood} ({sentiment:.2f})")

# Show past entries with sentiment graph
if st.session_state.history:
    st.markdown("## 📊 Mood Over Time")
    dates = [h["date"] for h in st.session_state.history]
    scores = [h["sentiment"] for h in st.session_state.history]

    fig, ax = plt.subplots()
    ax.plot(dates, scores, marker='o', linestyle='-')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    ax.set_title("Mood Trend")
    st.pyplot(fig)
