import streamlit as st
import datetime
import matplotlib.pyplot as plt
import google.generativeai as genai
from textblob import TextBlob
import pandas as pd
import base64
from wordcloud import WordCloud
import random
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import re
import plotly.express as px
import plotly.graph_objects as go
import os
import json
import pytz
from io import BytesIO
from PIL import Image
import numpy as np
from collections import Counter
import time
import hashlib

# Download necessary NLTK data
# Download necessary NLTK data
try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('tokenizers/punkt_tab')  # Added this check
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('punkt')
    nltk.download('punkt_tab')  # Added this download
    nltk.download('stopwords')

# App title and configuration
st.set_page_config(page_title="Daily AI Journal", layout="wide")

# Initialize session state variables
if "history" not in st.session_state:
    st.session_state.history = []
if "themes" not in st.session_state:
    st.session_state.themes = {}
if "goals" not in st.session_state:
    st.session_state.goals = []
if "reminders" not in st.session_state:
    st.session_state.reminders = []
if "custom_prompts" not in st.session_state:
    st.session_state.custom_prompts = {}
if "current_page" not in st.session_state:
    st.session_state.current_page = "journal"
if "user_data" not in st.session_state:
    st.session_state.user_data = {"name": "", "timezone": "UTC"}
if "streak" not in st.session_state:
    st.session_state.streak = {"current": 0, "last_entry": None}

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 42px;
        font-weight: bold;
        color: #4B0082;
        text-align: center;
        margin-bottom: 20px;
    }
    .subheader {
        font-size: 24px;
        font-weight: 500;
        color: #6A5ACD;
        margin-bottom: 15px;
    }
    .card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
        margin-bottom: 20px;
        background-color: #f8f9fa;
    }
    .streak-counter {
        font-size: 28px;
        font-weight: bold;
        color: #FF8C00;
        text-align: center;
    }
    .goal-item {
        padding: 10px;
        border-radius: 5px;
        margin: 5px 0;
        background-color: #e6f3ff;
    }
    .theme-tag {
        display: inline-block;
        padding: 5px 10px;
        border-radius: 15px;
        margin: 2px;
        font-size: 12px;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Helper Functions
def save_data_to_file():
    """Save session state data to a file"""
    data = {
        "history": st.session_state.history,
        "themes": st.session_state.themes,
        "goals": st.session_state.goals,
        "reminders": st.session_state.reminders,
        "custom_prompts": st.session_state.custom_prompts,
        "user_data": st.session_state.user_data,
        "streak": st.session_state.streak
    }
    
    # Create a folder if it doesn't exist
    if not os.path.exists("journal_data"):
        os.makedirs("journal_data")
        
    # Save to file
    user_hash = hashlib.md5(st.session_state.user_data["name"].encode()).hexdigest()
    with open(f"journal_data/{user_hash}.json", "w") as f:
        json.dump(data, f)

def load_data_from_file():
    """Load session state data from a file"""
    if not st.session_state.user_data["name"]:
        return False
        
    user_hash = hashlib.md5(st.session_state.user_data["name"].encode()).hexdigest()
    file_path = f"journal_data/{user_hash}.json"
    
    if os.path.exists(file_path):
        try:
            with open(file_path, "r") as f:
                data = json.load(f)
                st.session_state.history = data.get("history", [])
                st.session_state.themes = data.get("themes", {})
                st.session_state.goals = data.get("goals", [])
                st.session_state.reminders = data.get("reminders", [])
                st.session_state.custom_prompts = data.get("custom_prompts", {})
                st.session_state.user_data = data.get("user_data", {"name": "", "timezone": "UTC"})
                st.session_state.streak = data.get("streak", {"current": 0, "last_entry": None})
            return True
        except:
            return False
    return False

def build_prompt(entry, mode, custom_context=""):
    """Build prompt for AI based on selected mode"""
    mode_instructions = {
        "Coach": "Reply like a motivational coach. Give positive reinforcement, practical tips, and encourage action.",
        "Mentor": "Reply like a wise mentor. Provide insightful reflections, thought-provoking questions, and guidance.",
        "Friend": "Reply like a supportive best friend. Be kind, empathetic, and emotionally engaging.",
        "Therapist": "Reply like a professional therapist. Help identify patterns, provide gentle insights, and offer coping strategies.",
        "Future Self": "Reply as if you are the user's future self, looking back on this moment with wisdom and perspective.",
    }
    
    user_name = st.session_state.user_data["name"] if st.session_state.user_data["name"] else "User"
    
    # Add journal history context if available
    history_context = ""
    if len(st.session_state.history) > 3:
        recent_entries = st.session_state.history[-3:]
        history_context = "Recent journal context:\n"
        for h in recent_entries:
            history_context += f"Date: {h['date']}\nEntry: {h['entry'][:100]}...\n\n"
    
    # Add goals context if available
    goals_context = ""
    if st.session_state.goals:
        goals_context = "User's current goals:\n"
        for g in st.session_state.goals[:3]:
            goals_context += f"- {g['description']}\n"
    
    # Build the final prompt
    prompt = f"""You are a {mode.lower()} AI. {user_name} has shared their personal journal entry for today. 
Your job is to respond supportively and reflectively in a tone matching the role of a {mode.lower()}.

{mode_instructions.get(mode, mode_instructions["Friend"])}

{history_context}
{goals_context}
{custom_context}

Journal Entry:
{entry}

Your Response (keep it positive, thoughtful, and personalized to {user_name}):\n"""

    return prompt

def get_mood(sentiment):
    """Convert sentiment score to mood label"""
    if sentiment <= -0.7:
        return "😡 Very Angry"
    elif sentiment <= -0.5:
        return "😠 Angry"
    elif sentiment <= -0.3:
        return "😟 Sad"
    elif sentiment <= -0.1:
        return "😕 Slightly Negative"
    elif sentiment >= 0.7:
        return "🤩 Ecstatic"
    elif sentiment >= 0.5:
        return "😁 Very Positive"
    elif sentiment >= 0.3:
        return "😊 Happy"
    elif sentiment >= 0.1:
        return "🙂 Slightly Positive"
    else:
        return "😐 Neutral"

def get_random_tip():
    """Generate a random mindfulness or productivity tip"""
    tips = [
        "Take a 5-minute mindful breathing break.",
        "Write down 3 things you're grateful for.",
        "Go for a short walk to refresh your mind.",
        "Break big tasks into smaller, manageable chunks.",
        "Speak to yourself kindly. You deserve it.",
        "Drink a glass of water and notice how it makes you feel.",
        "Stretch your body for 2 minutes.",
        "Set a timer for 25 minutes of focused work.",
        "Text someone you care about just to say hello.",
        "Organize one small area of your workspace.",
        "Listen to one song with your full attention.",
        "Write down one worry and one possible solution.",
        "Practice the 4-7-8 breathing technique (inhale for 4, hold for 7, exhale for 8).",
        "Visualize your day going perfectly for 60 seconds.",
        "Put your phone in another room for one hour."
    ]
    return random.choice(tips)

def extract_keywords(text, num=5):
    """Extract key themes/topics from text"""
    # Tokenize and clean text
    stop_words = set(stopwords.words('english'))
    word_tokens = word_tokenize(text.lower())
    
    # Remove stopwords and non-alphabetic tokens
    filtered_words = [w for w in word_tokens if w not in stop_words and w.isalpha() and len(w) > 3]
    
    # Count word frequencies
    word_freq = Counter(filtered_words)
    
    # Return top N keywords
    return [word for word, freq in word_freq.most_common(num)]

def update_streak():
    """Update user's journaling streak"""
    today = datetime.date.today().isoformat()
    last_entry = st.session_state.streak.get("last_entry")
    
    if not last_entry:
        # First entry
        st.session_state.streak["current"] = 1
        st.session_state.streak["last_entry"] = today
        return
        
    # Convert to datetime objects
    last_date = datetime.date.fromisoformat(last_entry)
    today_date = datetime.date.today()
    
    # Calculate difference
    diff = (today_date - last_date).days
    
    if diff == 1:
        # Consecutive day
        st.session_state.streak["current"] += 1
        st.session_state.streak["last_entry"] = today
    elif diff == 0:
        # Same day, no change
        pass
    else:
        # Streak broken
        st.session_state.streak["current"] = 1
        st.session_state.streak["last_entry"] = today
        
def get_image_from_mood(sentiment):
    """Generate a simple mood visualization"""
    # Choose color based on sentiment
    if sentiment <= -0.5:
        color = (220, 50, 50)  # Red
    elif sentiment <= -0.2:
        color = (220, 150, 50)  # Orange
    elif sentiment >= 0.5:
        color = (50, 180, 50)  # Green
    elif sentiment >= 0.2:
        color = (150, 180, 50)  # Yellow-Green
    else:
        color = (200, 200, 200)  # Gray
        
    # Create a simple gradient image
    width, height = 200, 100
    image = Image.new('RGB', (width, height), color=(255, 255, 255))
    pixels = image.load()
    
    for i in range(width):
        for j in range(height):
            distance = (i / width)
            r = int(255 - distance * (255 - color[0]))
            g = int(255 - distance * (255 - color[1]))
            b = int(255 - distance * (255 - color[2]))
            pixels[i, j] = (r, g, b)
            
    return image

def analyze_journal_for_ai(entry):
    """Perform basic analysis for AI insights"""
    # Length analysis
    word_count = len(entry.split())
    complexity = len(set(entry.split())) / max(1, word_count)  # Unique words ratio
    
    # Sentiment
    sentiment = TextBlob(entry).sentiment.polarity
    
    # Keywords
    keywords = extract_keywords(entry)
    
    return {
        "word_count": word_count,
        "complexity": complexity,
        "sentiment": sentiment,
        "keywords": keywords
    }

# Navigation
def navigation():
    st.markdown('<div class="subheader">Navigation</div>', unsafe_allow_html=True)
    cols = st.columns(5)
    
    if cols[0].button("📝 Journal", use_container_width=True):
        st.session_state.current_page = "journal"
        st.rerun()
    
    if cols[1].button("📊 Analytics", use_container_width=True):
        st.session_state.current_page = "analytics"
        st.rerun()
        
    if cols[2].button("🎯 Goals", use_container_width=True):
        st.session_state.current_page = "goals"
        st.rerun()
    
    if cols[3].button("⚙️ Settings", use_container_width=True):
        st.session_state.current_page = "settings"
        st.rerun()
        
    if cols[4].button("❓ Help", use_container_width=True):
        st.session_state.current_page = "help"
        st.rerun()
    
    st.markdown("<hr>", unsafe_allow_html=True)

# Login/User Page
def login_page():
    st.markdown('<div class="main-header">🔐 Welcome to AI Journal</div>', unsafe_allow_html=True)
    
    user_name = st.text_input("Your Name (for personalization)")
    timezone = st.selectbox("Your Timezone", pytz.common_timezones, index=pytz.common_timezones.index("UTC"))
    
    if st.button("Start Journaling"):
        if user_name:
            st.session_state.user_data["name"] = user_name
            st.session_state.user_data["timezone"] = timezone
            
            # Try to load existing data
            if load_data_from_file():
                st.success(f"Welcome back, {user_name}! Your journal data has been loaded.")
            else:
                st.success(f"Welcome, {user_name}! Your new journal is ready.")
                
            st.session_state.current_page = "journal"
            st.rerun()
        else:
            st.error("Please enter your name to continue.")

# Journal Page
def journal_page():
    # Display header with user name if available
    user_name = st.session_state.user_data.get("name", "")
    if user_name:
        st.markdown(f'<div class="main-header">📝 {user_name}\'s AI Journal</div>', unsafe_allow_html=True)
    else:
        st.markdown('<div class="main-header">📝 Daily AI Journal</div>', unsafe_allow_html=True)

    # Gemini API Key Input
    api_key = st.text_input("Enter your Gemini API Key", type="password", key="api_key")
    if not api_key:
        st.warning("Please enter your Gemini API key to continue.")
        st.stop()

    # Configure Gemini API
    genai.configure(api_key=api_key)
    model = genai.GenerativeModel("gemini-2.0-flash")

    # Display streak
    if st.session_state.streak["current"] > 0:
        st.markdown(f'<div class="streak-counter">🔥 {st.session_state.streak["current"]} Day Streak</div>', unsafe_allow_html=True)

    # Mode selection
    col1, col2 = st.columns([3, 1])
    with col1:
        mode = st.selectbox("Choose AI Role:", ["Coach", "Mentor", "Friend", "Therapist", "Future Self"])
    
    with col2:
        if st.session_state.custom_prompts:
            use_custom = st.checkbox("Use custom prompt")
        else:
            use_custom = False
    
    # Custom prompt selection
    custom_context = ""
    if use_custom and st.session_state.custom_prompts:
        prompt_name = st.selectbox("Select custom prompt:", list(st.session_state.custom_prompts.keys()))
        custom_context = st.session_state.custom_prompts[prompt_name]
        st.info(f"Using custom prompt: {prompt_name}")

    # Journal Input
    entry = st.text_area("Your Journal Entry", height=250)
    
    # Tags
    tags = st.multiselect("Add tags:", 
                          options=list(st.session_state.themes.keys()) if st.session_state.themes else [], 
                          placeholder="Select or create tags")
    
    new_tag = st.text_input("Create new tag:")
    if new_tag and new_tag not in st.session_state.themes:
        # Generate a random color for the tag
        r = random.randint(100, 200)
        g = random.randint(100, 200)
        b = random.randint(100, 200)
        color = f"rgb({r},{g},{b})"
        
        st.session_state.themes[new_tag] = color
        tags.append(new_tag)
        st.success(f"Added new tag: {new_tag}")

    # Submission
    if st.button("Reflect with AI"):
        if not entry:
            st.error("Please write a journal entry before submitting.")
            return
            
        with st.spinner("Thinking..."):
            # Update streak
            update_streak()
            
            # Build prompt and get AI response
            prompt = build_prompt(entry, mode, custom_context)
            response = model.generate_content(prompt)
            
            # Analyze sentiment and extract keywords
            sentiment = TextBlob(entry).sentiment.polarity
            keywords = extract_keywords(entry)
            mood = get_mood(sentiment)
            
            # Additional analysis
            entry_stats = analyze_journal_for_ai(entry)
            
            # Save the reflection
            reflection = {
                "date": datetime.date.today().isoformat(),
                "entry": entry,
                "response": response.text,
                "sentiment": sentiment,
                "keywords": keywords,
                "tags": tags,
                "mode": mode,
                "stats": entry_stats
            }
            st.session_state.history.append(reflection)
            
            # Save data to file
            save_data_to_file()
            
            # Display response
            st.markdown('<div class="card">', unsafe_allow_html=True)
            st.markdown("### 🤖 AI's Response")
            st.write(response.text)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # Display analytics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.markdown(f"### 🧠 Sentiment: {mood}")
                st.progress((sentiment + 1) / 2)  # Convert -1,1 to 0,1 range
            
            with col2:
                st.markdown("### 🔑 Key Themes")
                if keywords:
                    st.write(", ".join(keywords))
                else:
                    st.write("No significant themes detected")
            
            with col3:
                st.markdown(f"### 💡 Tip for You")
                st.info(get_random_tip())
            
            # Display mood image
            mood_img = get_image_from_mood(sentiment)
            buf = BytesIO()
            mood_img.save(buf, format="PNG")
            st.image(buf.getvalue(), caption=f"Mood: {mood} ({sentiment:.2f})")

# Analytics Page
def analytics_page():
    st.markdown('<div class="main-header">📊 Journal Analytics</div>', unsafe_allow_html=True)
    
    if not st.session_state.history:
        st.info("Start journaling to see your analytics.")
        return
    
    # Convert history to DataFrame for easier analysis
    df = pd.DataFrame([
        {
            "date": h["date"],
            "sentiment": h["sentiment"],
            "entry": h["entry"],
            "word_count": len(h["entry"].split()),
            "tags": ", ".join(h.get("tags", [])),
            "mode": h.get("mode", "Not specified")
        } for h in st.session_state.history
    ])
    
    # Convert date to datetime
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date")
    
    # Mood over time
    st.markdown('<div class="subheader">Mood Trends</div>', unsafe_allow_html=True)
    
    fig = px.line(df, x="date", y="sentiment", 
                  title="Sentiment Over Time",
                  labels={"sentiment": "Mood Score", "date": "Date"},
                  markers=True)
    
    fig.add_hline(y=0, line_dash="dash", line_color="gray")
    fig.update_layout(
        xaxis_title="Date",
        yaxis_title="Sentiment Score (-1 to 1)",
        yaxis=dict(range=[-1, 1])
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Average sentiment by day of week
    st.markdown('<div class="subheader">Sentiment by Day of Week</div>', unsafe_allow_html=True)
    
    df["day_of_week"] = df["date"].dt.day_name()
    day_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    
    day_sentiments = df.groupby("day_of_week")["sentiment"].mean().reindex(day_order)
    
    fig = px.bar(
        x=day_sentiments.index, 
        y=day_sentiments.values,
        labels={"x": "Day of Week", "y": "Average Sentiment"},
        color=day_sentiments.values,
        color_continuous_scale="RdYlGn"
    )
    fig.update_layout(title="Average Mood by Day of Week")
    st.plotly_chart(fig, use_container_width=True)
    
    # Word count trends
    st.markdown('<div class="subheader">Journal Entry Length</div>', unsafe_allow_html=True)
    
    fig = px.bar(
        df, 
        x="date", 
        y="word_count",
        title="Journal Entry Length Over Time"
    )
    st.plotly_chart(fig, use_container_width=True)
    
    # Most used tags
    if any("tags" in h for h in st.session_state.history):
        st.markdown('<div class="subheader">Most Used Tags</div>', unsafe_allow_html=True)
        
        all_tags = []
        for h in st.session_state.history:
            if "tags" in h and h["tags"]:
                all_tags.extend(h["tags"])
        
        if all_tags:
            tag_counts = Counter(all_tags)
            
            fig = px.pie(
                values=list(tag_counts.values()), 
                names=list(tag_counts.keys()),
                title="Journal Tags Distribution"
            )
            st.plotly_chart(fig, use_container_width=True)
    
    # Word cloud of all entries
    st.markdown('<div class="subheader">Journal Word Cloud</div>', unsafe_allow_html=True)
    
    all_entries = " ".join([h["entry"] for h in st.session_state.history])
    
    # Generate word cloud
    wordcloud = WordCloud(
        width=800, 
        height=400, 
        background_color='white',
        colormap='viridis',
        max_words=100
    ).generate(all_entries)
    
    # Display word cloud
    fig_wc, ax_wc = plt.subplots(figsize=(10, 5))
    ax_wc.imshow(wordcloud, interpolation='bilinear')
    ax_wc.axis('off')
    st.pyplot(fig_wc)
    
    # Tag-based sentiment analysis
    if st.session_state.themes:
        st.markdown('<div class="subheader">Tag-based Sentiment Analysis</div>', unsafe_allow_html=True)
        
        tag_sentiments = {}
        tag_counts = {}
        
        for h in st.session_state.history:
            if "tags" in h and h["tags"]:
                for tag in h["tags"]:
                    if tag not in tag_sentiments:
                        tag_sentiments[tag] = []
                    tag_sentiments[tag].append(h["sentiment"])
        
        # Calculate average sentiment by tag
        avg_tag_sentiments = {tag: sum(sentiments)/len(sentiments) for tag, sentiments in tag_sentiments.items()}
        
        if avg_tag_sentiments:
            # Sort by sentiment
            sorted_tags = sorted(avg_tag_sentiments.items(), key=lambda x: x[1])
            
            fig = px.bar(
                x=[tag for tag, _ in sorted_tags],
                y=[sentiment for _, sentiment in sorted_tags],
                labels={"x": "Tag", "y": "Average Sentiment"},
                color=[sentiment for _, sentiment in sorted_tags],
                color_continuous_scale="RdYlGn"
            )
            fig.update_layout(title="Average Mood by Tag")
            st.plotly_chart(fig, use_container_width=True)

# Goals Page
def goals_page():
    st.markdown('<div class="main-header">🎯 Goals & Intentions</div>', unsafe_allow_html=True)
    
    # Add new goal
    with st.expander("Add New Goal or Intention", expanded=True):
        goal_description = st.text_area("Describe your goal or intention:", placeholder="I want to...")
        goal_deadline = st.date_input("Target date (optional):", min_value=datetime.date.today())
        goal_category = st.selectbox("Category:", ["Personal", "Work", "Health", "Learning", "Relationships", "Other"])
        
        if st.button("Add Goal"):
            if goal_description:
                goal_id = len(st.session_state.goals)
                new_goal = {
                    "id": goal_id,
                    "description": goal_description,
                    "deadline": goal_deadline.isoformat(),
                    "category": goal_category,
                    "created": datetime.date.today().isoformat(),
                    "completed": False,
                    "reflections": []
                }
                st.session_state.goals.append(new_goal)
                save_data_to_file()
                st.success("Goal added successfully!")
                st.rerun()
            else:
                st.error("Please enter a goal description.")
    
    # Display active goals
    st.markdown('<div class="subheader">Active Goals</div>', unsafe_allow_html=True)
    
    active_goals = [g for g in st.session_state.goals if not g["completed"]]
    
    if not active_goals:
        st.info("No active goals. Add one above to get started!")
    
    for i, goal in enumerate(active_goals):
        with st.container():
            col1, col2, col3 = st.columns([3, 1, 1])
            
            with col1:
                st.markdown(f'<div class="goal-item">{goal["description"]}</div>', unsafe_allow_html=True)
                st.caption(f"Category: {goal['category']} | Due: {goal['deadline']} | Created: {goal['created']}")
            
            with col2:
                if st.button(f"✓ Complete", key=f"complete_{i}"):
                    st.session_state.goals[goal["id"]]["completed"] = True
                    save_data_to_file()
                    st.success("Goal marked as complete!")
                    st.rerun()
            
            with col3:
                if st.button(f"🔍 Reflect", key=f"reflect_{i}"):
                    # Add reflection functionality here
                    st.session_state.current_goal_id = goal["id"]
                    st.session_state.current_page = "goal_reflection"
                    st.rerun()
        
        st.markdown("<hr>", unsafe_allow_html=True)
    
    # Display completed goals
    completed_goals = [g for g in st.session_state.goals if g["completed"]]
    
    if completed_goals:
        with st.expander("Completed Goals", expanded=False):
            for goal in completed_goals:
                st.markdown(f"✓ **{goal['description']}** _(Completed)_")
                st.caption(f"Category: {goal['category']} | Created: {goal['created']}")
                st.markdown("---")

# Goal Reflection Page
def goal_reflection_page():
    goal_id = st.session_state.current_goal_id
    goal = next((g for g in st.session_state.goals if g["id"] == goal_id), None)
    
    if not goal:
        st.error("Goal not found.")
        st.session_state.current_page = "goals"
        st.rerun()
        return
    
    st.markdown(f'<div class="main-header">🔍 Goal Reflection</div>', unsafe_allow_html=True)
    st.markdown(f'<div class="subheader">{goal["description"]}</div>', unsafe_allow_html=True)
    
    reflection = st.text_area("How are you progressing with this goal? What's working? What challenges are you facing?", 
                             height=150)
    
    if st.button("Save Reflection"):
        if reflection:
            new_reflection = {
                "date": datetime.date.today().isoformat(),
                "text": reflection
            }
            
            if "reflections" not in st.session_state.goals[goal_id]:
                st.session_state.goals[goal_id]["reflections"] = []
                
            st.session_state.goals[goal_id]["reflections"].append(new_reflection)
            save_data_to_file()
            st.success("Reflection added!")
        else:
            st.error("Please write a reflection before saving.")
    
    if st.button("Back to Goals"):
        st.session_state.current_page = "goals"
        st.rerun()
    
    # Show previous reflections
    if "reflections" in goal and goal["reflections"]:
        st.markdown("### Previous Reflections")
        
        for r in goal["reflections"]:
            st.markdown(f"**{r['date']}**")
            st.markdown(f"> {r['text']}")
            st.markdown("---")

# Settings Page
def settings_page():
    st.markdown('<div class="main-header">⚙️ Settings</div>', unsafe_allow_html=True)
    
    # User settings
    st.markdown('<div class="subheader">User Settings</div>', unsafe_allow_html=True)
    
    user_name = st.text_input("Your Name:", value=st.session_state.user_data.get("name", ""))
    timezone = st.selectbox("Your Timezone:", 
                           options=pytz.common_timezones, 
                           index=pytz.common_timezones.index(st.session_state.user_data.get("timezone", "UTC")))
    
    if st.button("Save User Settings"):
        st.session_state.user_data["name"] = user_name
        st.session_state.user_data["timezone"] = timezone
        save_data_to_file()
        st.success("User settings saved!")
    
    # Custom prompts
    st.markdown('<div class="subheader">Custom Prompts</div>', unsafe_allow_html=True)
    st.markdown("Create custom prompts to guide AI responses in specific ways.")
    
    with st.expander("Create New Custom Prompt", expanded=False):
        prompt_name = st.text_input("Prompt Name:", key="new_prompt_name")
        prompt_content = st.text_area("Prompt Instructions:", 
                                     placeholder="Example: Focus on my career goals and provide advice for professional development.",
                                     height=150,
                                     key="new_prompt_content")
        
        if st.button("Add Custom Prompt"):
            if prompt_name and prompt_content:
                st.session_state.custom_prompts[prompt_name] = prompt_content
                save_data_to_file()
                st.success(f"Custom prompt '{prompt_name}' created!")
            else:
                st.error("Please enter both a name and content for your custom prompt.")
    
    # Display and manage existing prompts
    if st.session_state.custom_prompts:
        st.markdown("### Your Custom Prompts")
        
        for name, content in st.session_state.custom_prompts.items():
            with st.expander(f"Prompt: {name}", expanded=False):
                st.write(content)
                if st.button(f"Delete '{name}'", key=f"delete_{name}"):
                    del st.session_state.custom_prompts[name]
                    save_data_to_file()
                    st.success(f"Prompt '{name}' deleted!")
                    st.rerun()
    
    # Reminders
    st.markdown('<div class="subheader">Journal Reminders</div>', unsafe_allow_html=True)
    st.markdown("Set up reminders for your journaling practice (these will be stored locally).")
    
    reminder_enabled = st.checkbox("Enable Daily Reminder", 
                                  value=len(st.session_state.reminders) > 0)
    
    if reminder_enabled:
        reminder_time = st.time_input("Reminder Time:", 
                                     value=datetime.time(20, 0) if not st.session_state.reminders else 
                                     datetime.time.fromisoformat(st.session_state.reminders[0]["time"]))
        
        reminder_message = st.text_input("Reminder Message:", 
                                        value="Time for your daily journal reflection!" if not st.session_state.reminders else
                                        st.session_state.reminders[0]["message"])
        
        if st.button("Save Reminder Settings"):
            if reminder_enabled:
                st.session_state.reminders = [{
                    "time": reminder_time.isoformat(),
                    "message": reminder_message,
                    "enabled": True
                }]
            else:
                st.session_state.reminders = []
                
            save_data_to_file()
            st.success("Reminder settings saved!")
    
    # Export/Import
    st.markdown('<div class="subheader">Data Management</div>', unsafe_allow_html=True)
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.session_state.history:
            # Create a DataFrame and convert to CSV
            df = pd.DataFrame(st.session_state.history)
            csv = df.to_csv(index=False).encode("utf-8")
            b64 = base64.b64encode(csv).decode()
            
            st.markdown(f'<a href="data:file/csv;base64,{b64}" download="journal_history.csv" class="button">📥 Export Journal History</a>', 
                       unsafe_allow_html=True)
    
    with col2:
        if st.button("🗑️ Clear All Data"):
            confirm = st.checkbox("I understand this will delete ALL my journal data")
            
            if confirm and st.button("Confirm Delete"):
                st.session_state.history = []
                st.session_state.themes = {}
                st.session_state.goals = []
                st.session_state.reminders = []
                st.session_state.custom_prompts = {}
                st.session_state.streak = {"current": 0, "last_entry": None}
                
                save_data_to_file()
                st.success("All data has been cleared.")
                st.rerun()

# Help Page
def help_page():
    st.markdown('<div class="main-header">❓ Help & Tips</div>', unsafe_allow_html=True)
    
    with st.expander("Getting Started", expanded=True):
        st.markdown("""
        ### Welcome to AI Journal!
        
        This app helps you maintain a daily journal with AI-powered reflections. Here's how to get started:
        
        1. **Enter your Gemini API key** in the Journal page
        2. **Write your daily entry** in the text area
        3. **Choose an AI role** that matches what you need today
        4. Click **Reflect with AI** to get your personalized response
        
        The AI will analyze your entry and provide thoughtful reflections based on your chosen role.
        """)
    
    with st.expander("Features Overview"):
        st.markdown("""
        ### Main Features
        
        - **AI Reflections**: Get personalized responses from different AI personas
        - **Mood Tracking**: Automatically analyze the sentiment of your entries
        - **Journal Analytics**: See patterns in your mood and writing over time
        - **Goal Setting**: Set and track personal goals alongside your journal
        - **Custom Prompts**: Create specific guidance for the AI responses
        - **Journaling Streaks**: Build consistency with a streak counter
        - **Tags**: Organize entries with custom tags for better tracking
        """)
    
    with st.expander("Journal Tips"):
        st.markdown("""
        ### Tips for Effective Journaling
        
        1. **Be consistent**: Try to write at the same time each day
        2. **Be honest**: Your journal is private, so express your true thoughts
        3. **Don't self-censor**: Write whatever comes to mind
        4. **Use prompts**: If you're stuck, try these starters:
           - What made me smile today?
           - What challenged me today?
           - What am I grateful for?
           - What did I learn today?
           - What's one thing I could have done better?
        5. **Review regularly**: Look back at past entries to see your growth
        """)
    
    with st.expander("Troubleshooting"):
        st.markdown("""
        ### Common Issues
        
        **API Key Not Working**
        - Make sure you've entered a valid Gemini API key
        - Check that you have an active Gemini account with API access
        
        **Data Not Saving**
        - Your data is saved locally. Make sure you're using the same browser and haven't cleared your cache
        - Export your data regularly for backup
        
        **AI Responses Not Helpful**
        - Try switching to a different AI role
        - Be more specific and detailed in your journal entries
        - Create a custom prompt to guide the AI's responses
        """)
    
    with st.expander("About the App"):
        st.markdown("""
        ### About AI Journal
        
        This app was created to help people reflect more deeply on their daily experiences through the power of AI. It combines traditional journaling with modern AI analysis to provide insights and patterns you might miss on your own.
        
        The app uses:
        - Google's Gemini AI for reflections
        - TextBlob for sentiment analysis
        - Streamlit for the interface
        - Local storage for data privacy
        
        All your journal data is stored locally on your device and is not shared with any third parties.
        """)

# Main application logic
def main():
    # Check if user is logged in
    if not st.session_state.user_data.get("name") and st.session_state.current_page != "login":
        st.session_state.current_page = "login"
    
    # Login page
    if st.session_state.current_page == "login":
        login_page()
        return
    
    # Show navigation for all other pages
    navigation()
    
    # Render current page
    if st.session_state.current_page == "journal":
        journal_page()
    elif st.session_state.current_page == "analytics":
        analytics_page()
    elif st.session_state.current_page == "goals":
        goals_page()
    elif st.session_state.current_page == "goal_reflection":
        goal_reflection_page()
    elif st.session_state.current_page == "settings":
        settings_page()
    elif st.session_state.current_page == "help":
        help_page()
    else:
        journal_page()  # Default to journal page

if __name__ == "__main__":
    main()
