# AI Journals

![Language](https://img.shields.io/badge/Language-Python-3776AB?style=flat-square) ![Stars](https://img.shields.io/github/stars/Devanik21/AI-journals-?style=flat-square&color=yellow) ![Forks](https://img.shields.io/github/forks/Devanik21/AI-journals-?style=flat-square&color=blue) ![Author](https://img.shields.io/badge/Author-Devanik21-black?style=flat-square&logo=github) ![Status](https://img.shields.io/badge/Status-Active-brightgreen?style=flat-square)

> AI-assisted journalling — intelligent prompts, mood tracking, pattern recognition, and reflective insights from your writing history.

---

**Topics:** `academic-ai` · `annotation` · `deep-learning` · `generative-ai` · `knowledge-management` · `large-language-models` · `nlp` · `note-taking` · `research-paper-management` · `text-summarization`

## Overview

AI Journals is a private, AI-augmented journalling application that transforms the solitary act of
keeping a journal into a richer reflective practice. Beyond providing a clean writing interface, it
generates contextually appropriate journal prompts tailored to the day, mood, and recent entry themes;
analyses writing patterns over time; identifies recurring emotional themes and cognitive patterns;
and generates periodic reflective synthesis reports — weekly or monthly summaries that surface insights
the writer might not notice entry by entry.

The privacy architecture is designed around local-first principles: by default, journal entries are
stored in an encrypted SQLite database on the local machine, never transmitted to cloud servers.
The AI analysis features can be configured to run either locally (with a small LLM via Ollama) or
via a cloud API — giving the user explicit control over whether their private writing ever leaves
their device.

The mood tracking system uses a combination of explicit emoji-based mood tagging and implicit
sentiment analysis of the entry text to build a longitudinal mood record. This record is visualised
as a calendar heatmap (GitHub contribution-style) and a time-series chart that reveals cycles,
correlations with external events (holidays, weekdays vs weekends), and the emotional trajectory
across weeks and months.

---

## Motivation

Journalling is one of the most evidence-backed practices for emotional processing, self-awareness,
and cognitive clarity. But the blank page problem — not knowing what to write — stops many people
before they start. And the insights buried in months of entries are nearly impossible to surface
manually. AI Journals was built to solve both problems: removing the blank page with intelligent
prompts, and making the accumulated wisdom of your own writing accessible through analysis.

---

## Architecture

```
Journal Entry (text + mood tag)
        │
  SQLite encrypted local storage (sqlcipher)
        │
  ┌─────────────────────────────────────────────┐
  │  Analysis Pipeline:                         │
  │  ├── Sentiment analysis (VADER)             │
  │  ├── Topic modelling (LDA / NMF)            │
  │  ├── Emotion classification (NRC lexicon)   │
  │  └── Pattern detection (temporal, thematic) │
  └─────────────────────────────────────────────┘
        │
  LLM: reflective synthesis + personalised prompts
        │
  Streamlit: journal editor + analytics dashboard
```

---

## Features

### AI Journal Prompts
Context-aware prompts generated based on time of day, day of week, recent entry themes, and current mood — ranging from gratitude and intention-setting in the morning to reflection and processing in the evening.

### Encrypted Local Storage
All entries stored in an SQLCipher-encrypted SQLite database on the local machine — no cloud dependency, no data leaving the device without explicit user action.

### Mood Tracking System
Emoji-based explicit mood tagging combined with automatic sentiment analysis of entry text, building a longitudinal emotional record with no extra effort.

### Calendar Heatmap Visualisation
GitHub-style contribution calendar heatmap of writing frequency and mood score by day, making consistency and emotional patterns immediately visible.

### Theme and Topic Analysis
LDA or NMF topic modelling across all entries to identify recurring themes, concerns, and preoccupations that span individual entries — revealing what the writer consistently thinks about.

### Weekly and Monthly Synthesis
LLM-generated reflective synthesis reports that summarise the period's key themes, emotional trajectory, notable events mentioned, and forward-looking intentions expressed.

### Writing Statistics
Word count trends, writing streak tracking, most active days/times, vocabulary richness (type-token ratio), and average entry length over time.

### Search and Retrieval
Full-text semantic search across all entries using TF-IDF or sentence-transformer embeddings, supporting natural language queries like 'when was I most anxious?' or 'entries about my career'.

---

## Tech Stack

| Library / Tool | Role | Why This Choice |
|---|---|---|
| **SQLite + SQLCipher** | Encrypted local storage | Privacy-preserving encrypted journal database |
| **VADER Sentiment** | Mood analysis | Valence-aware sentiment scoring of entry text |
| **scikit-learn LDA** | Topic modelling | Recurring theme extraction across entry corpus |
| **OpenAI/Ollama LLM** | Synthesis + prompts | Reflective reports and contextual prompt generation |
| **Streamlit** | Journal UI | Editor, mood picker, analytics dashboard |
| **Plotly** | Visualisation | Calendar heatmap, mood timeline, topic wordcloud |
| **sentence-transformers** | Semantic search | Entry embedding for natural language retrieval |

---

## Getting Started

### Prerequisites

- Python 3.9+ (or Node.js 18+ for TypeScript/JavaScript projects)
- A virtual environment manager (`venv`, `conda`, or equivalent)
- API keys as listed in the Configuration section

### Installation

```bash
git clone https://github.com/Devanik21/AI-journals-.git
cd AI-journals-
python -m venv venv && source venv/bin/activate
pip install streamlit vadersentiment scikit-learn sentence-transformers \
            plotly pandas openai python-dotenv
# Optional: local LLM for full privacy
# pip install ollama
echo 'OPENAI_API_KEY=sk-...' > .env  # or configure Ollama
echo 'DB_ENCRYPTION_KEY=your_passphrase' >> .env
streamlit run app.py
```

---

## Usage

```bash
# Launch journal
streamlit run app.py

# Export entries as Markdown archive
python export.py --format markdown --output journal_export/

# Generate monthly synthesis report
python synthesise.py --period 2026-02

# Search entries
python search.py --query 'feeling overwhelmed at work'
```

---

## Configuration

| Variable | Default | Description |
|---|---|---|
| `DB_ENCRYPTION_KEY` | `(required)` | Passphrase for SQLCipher database encryption |
| `LLM_BACKEND` | `openai` | LLM backend: openai, gemini, ollama |
| `OPENAI_API_KEY` | `(if using OpenAI)` | OpenAI API key |
| `SYNTHESIS_FREQUENCY` | `weekly` | Report generation frequency: daily, weekly, monthly |
| `PROMPT_STYLE` | `reflective` | Prompt style: reflective, gratitude, creative, goal-focused |

> Copy `.env.example` to `.env` and populate required values before running.

---

## Project Structure

```
AI-journals/
├── README.md
├── requirements.txt
├── app.py
└── ...
```

---

## Roadmap

- [ ] Voice journalling mode with Whisper transcription for audio entries
- [ ] Cognitive reframing suggestions for entries with persistently negative sentiment
- [ ] Integration with Apple Health / Google Fit for correlation of mood with sleep, exercise, and activity
- [ ] Collaborative couple/family journal mode with separate encryption keys per participant
- [ ] Retrospective letter generation: write a letter to your past self based on entry history

---

## Contributing

Contributions, issues, and suggestions are welcome.

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/your-idea`
3. Commit your changes: `git commit -m 'feat: add your idea'`
4. Push to your branch: `git push origin feature/your-idea`
5. Open a Pull Request with a clear description

Please follow conventional commit messages and add documentation for new features.

---

## Notes

Journal entries are private and sensitive data. By default, NO data is transmitted to cloud services. The cloud LLM features (synthesis reports, prompts) transmit only the most recent entry or a topic summary — never the full journal history. Review the privacy configuration before enabling cloud LLM features.

---

## Author

**Devanik Debnath**  
B.Tech, Electronics & Communication Engineering  
National Institute of Technology Agartala

[![GitHub](https://img.shields.io/badge/GitHub-Devanik21-black?style=flat-square&logo=github)](https://github.com/Devanik21)
[![LinkedIn](https://img.shields.io/badge/LinkedIn-devanik-blue?style=flat-square&logo=linkedin)](https://www.linkedin.com/in/devanik/)

---

## License

This project is open source and available under the [MIT License](LICENSE).

---

*Built with curiosity, depth, and care — because good projects deserve good documentation.*
