import streamlit as st
import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
from textblob import TextBlob
import pandas as pd
import base64
from wordcloud import WordCloud
import random

# App title
st.set_page_config(page_title="Daily AI Journal", layout="wide")
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

# Tip generator
def get_random_tip():
    tips = [
        "Take a 5-minute mindful breathing break.",
        "Write down 3 things you're grateful for.",
        "Go for a short walk to refresh your mind.",
        "Break big tasks into smaller, manageable chunks.",
        "Speak to yourself kindly. You deserve it."
    ]
    return random.choice(tips)

# Load past history
if "history" not in st.session_state:
    st.session_state.history = []

# Sidebar features
with st.sidebar:
    st.header("🔧 Journal Tools")
    if st.button("🗑️ Clear History"):
        st.session_state.history.clear()
        st.success("History cleared!")

    if st.session_state.history:
        df = pd.DataFrame(st.session_state.history)
        csv = df.to_csv(index=False).encode("utf-8")
        b64 = base64.b64encode(csv).decode()
        st.markdown(f'<a href="data:file/csv;base64,{b64}" download="journal_history.csv">📥 Download History</a>', unsafe_allow_html=True)

    if st.button("🌥️ Show Word Cloud") and st.session_state.history:
        all_entries = " ".join([h["entry"] for h in st.session_state.history])
        wordcloud = WordCloud(width=800, height=400, background_color='white').generate(all_entries)
        fig_wc, ax_wc = plt.subplots(figsize=(8, 4))
        ax_wc.imshow(wordcloud, interpolation='bilinear')
        ax_wc.axis('off')
        st.pyplot(fig_wc)

# Mode selection
mode = st.selectbox("Choose AI Role:", ["Coach", "Mentor", "Friend"])

# Journal Input
entry = st.text_area("Your Journal Entry", height=250)

# Submission
if st.button("Reflect with AI"):
    with st.spinner("Thinking..."):
        prompt = build_prompt(entry, mode)
        response = model.generate_content(prompt)
        sentiment = TextBlob(entry).sentiment.polarity
        mood = get_mood(sentiment)

        reflection = {
            "date": datetime.date.today().isoformat(),
            "entry": entry,
            "response": response.text,
            "sentiment": sentiment
        }
        st.session_state.history.append(reflection)

        st.markdown("### 🤖 AI's Response")
        st.write(response.text)

        st.markdown(f"### 🧠 Sentiment: {mood} ({sentiment:.2f})")
        st.markdown(f"### 💡 Tip for You: {get_random_tip()}")

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

    # Show history table
    st.markdown("## 📖 Journal History")
    for h in st.session_state.history[::-1]:
        with st.expander(f"{h['date']} — {get_mood(h['sentiment'])}"):
            st.markdown(f"**Journal Entry:**\n{h['entry']}")
            st.markdown(f"**AI's Response:**\n{h['response']}")
            st.markdown(f"**Sentiment Score:** {h['sentiment']:.2f}")
