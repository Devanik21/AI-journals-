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
model = genai.GenerativeModel("gemini-2.0-flash")

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

# Mood analyzer
def get_mood(sentiment):
    if sentiment <= -0.5:
        return "😡 Angry"
    elif sentiment <= -0.2:
        return "😟 Negative"
    elif sentiment >= 0.5:
        return "😁 Very Positive"
    elif sentiment >= 0.2:
        return "😊 Positive"
    else:
        return "😐 Neutral"

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
        mood = get_mood(sentiment)

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
    ax.plot(dates, scores, marker='o', linestyle='-', color='purple')
    ax.axhline(0, color='gray', linestyle='--')
    ax.set_ylabel("Sentiment Score")
    ax.set_xlabel("Date")
    ax.set_title("Mood Trend")
    plt.xticks(rotation=45)
    st.pyplot(fig)

    # Optional: Show last 3 journal responses
    st.markdown("## 🕰️ Recent Reflections")
    for h in st.session_state.history[-3:][::-1]:
        st.markdown(f"**{h['date']}** — *{get_mood(h['sentiment'])}*\n\n**Entry:** {h['entry']}\n\n**AI Response:** {h['response']}")
