import os
import smtplib
import time
import threading
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from datetime import datetime

import feedparser
import requests
import re
import html as _html
import textwrap
import streamlit as st
from dotenv import load_dotenv
import schedule

# ------------------ LOAD ENV ------------------ #
load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")

EMAIL_HOST = os.getenv("EMAIL_HOST", "smtp.gmail.com")
EMAIL_PORT = int(os.getenv("EMAIL_PORT", "587"))
EMAIL_USER = os.getenv("EMAIL_USER")
EMAIL_PASSWORD = os.getenv("EMAIL_PASSWORD")

# ------------------ CONFIG ------------------ #
RSS_FEEDS = [
    "https://www.theverge.com/rss/index.xml",
    "https://www.technologyreview.com/feed/",
    "https://feeds.arstechnica.com/arstechnica/technology",
    "https://feeds.bloomberg.com/markets/news.rss",
    "https://feeds.wired.com/wired/index",
    "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "https://feeds.reuters.com/reuters/technologyNews",
    "https://www.engadget.com/feed.xml",
    "https://techcrunch.com/feed/",
    "https://feeds2.bloomberg.com/technology/news.rss",
]

AI_KEYWORDS = ["ai", "artificial intelligence", "machine learning", "ml", "llm", "openai", "deep learning"]


# ------------------ STEP 1: FETCH NEWS ------------------ #
def fetch_ai_news(max_items: int = 15):
    """Fetch recent AI-related news from RSS feeds and return a list of dicts."""
    articles = []

    for feed_url in RSS_FEEDS:
        try:
            feed = feedparser.parse(feed_url)
        except Exception as e:
            print(f"Error parsing feed {feed_url}: {e}")
            continue

        for entry in feed.entries:
            title = entry.get("title", "")
            summary = entry.get("summary", "")
            link = entry.get("link", "")

            text = f"{title} {summary}".lower()
            if any(kw in text for kw in AI_KEYWORDS):
                articles.append(
                    {
                        "title": title,
                        "summary": summary,
                        "link": link,
                        "source": feed.feed.get("title", "Unknown Source"),
                    }
                )

    return articles[:max_items]


# ------------------ STEP 2: CALL GROQ LLM ------------------ #
def summarize_with_groq(articles):
    """Send the collected articles to Groq and get a concise digest."""
    if not GROQ_API_KEY:
        raise ValueError("GROQ_API_KEY is not set in .env")

    if not articles:
        return "No recent AI-related articles were found from the configured sources."

    # Build text for the LLM
    articles_text = ""
    for i, art in enumerate(articles, start=1):
        articles_text += (
            f"{i}. Title: {art['title']}\n"
            f"   Source: {art['source']}\n"
            f"   Summary: {art['summary']}\n"
            f"   Link: {art['link']}\n\n"
        )

    system_prompt = (
        "You are an AI news assistant. Your job is to read the list of recent technology news "
        "and select only the most important AI-related updates. Focus on LLMs, AI models, tools, "
        "regulation, big product launches, and breakthroughs.\n\n"
        "Write a concise email-style summary with:\n"
        "- A short intro (1–2 lines)\n"
        "- 3 to 7 bullet points of key AI updates\n"
        "- Each bullet should have: what happened, why it matters, and (if useful) the company/model name.\n"
        "- At the end, add a section 'Links' listing the selected articles' titles with their URLs.\n"
    )

    user_prompt = (
        "Here are some recent articles. Pick the most important AI updates and create an email digest:\n\n"
        f"{articles_text}"
    )

    url = "https://api.groq.com/openai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {GROQ_API_KEY}",
        "Content-Type": "application/json",
    }

    payload = {
        "model": "llama-3.1-8b-instant",  # You can change to another Groq model
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "temperature": 0.5,
        "max_tokens": 800,
    }

    response = requests.post(url, headers=headers, json=payload, timeout=60)
    response.raise_for_status()
    data = response.json()

    content = data["choices"][0]["message"]["content"]
    return content


# ------------------ FORMATTER (local) ------------------ #
def _clean_text(text: str) -> str:
    """Strip HTML, unescape entities and collapse whitespace."""
    if not text:
        return ""
    # remove HTML tags
    text = re.sub(r"<[^>]+>", "", text)
    # unescape html entities
    text = _html.unescape(text)
    # collapse whitespace
    text = re.sub(r"\s+", " ", text).strip()
    return text


def _first_sentences(text: str, max_sentences: int = 2) -> str:
    text = _clean_text(text)
    # split on sentence endings
    parts = re.split(r'(?<=[.!?])\s+', text)
    return " ".join(parts[:max_sentences]).strip()


def _choose_emoji(title: str, summary: str) -> str:
    t = (title + " " + summary).lower()
    if "robot" in t or "humanoid" in t or "robotics" in t:
        return "🔥"
    if "index" in t or "metrics" in t or "hype" in t or "data" in t or "analytics" in t:
        return "📊"
    if "protein" in t or "alphafold" in t or "bio" in t or "biotech" in t:
        return "🧬"
    if "privacy" in t or "consent" in t or "data" in t or "security" in t:
        return "🔐"
    if "microsoft" in t or "windows" in t or "pc" in t or "agent" in t:
        return "💻"
    return "🔎"


def create_formatted_digest(articles: list) -> tuple:
    """Create a nicely formatted digest (subject, body) from article list.

    Returns (subject, body_text)
    """
    if not articles:
        subject = "AI Updates Digest — Your Daily Briefing"
        body = "Hello Reader,\n\nNo recent AI-related articles were found."
        return subject, body

    # pick top 5 articles
    selected = articles[:5]

    # Build subject: use key topics from first two titles
    def _short_title(t):
        t = _clean_text(t)
        return textwrap.shorten(t, width=60, placeholder="...")

    first_titles = ", ".join([_short_title(a.get("title", "")) for a in selected[:3]])
    subject = f"🚀 Top AI Developments: {first_titles}"

    # Header and intro
    lines = []
    lines.append("AI Updates Digest — Your Daily Briefing")
    lines.append(f"Subject: {subject}")
    lines.append("")
    lines.append("Hello Reader,")
    lines.append("")
    lines.append("Here’s your crisp, high-value rundown of the most important AI developments this week — curated, structured, and enhanced for clarity.")
    lines.append("")

    # Add enumerated bullets
    for i, art in enumerate(selected, start=1):
        title = art.get("title", "No title")
        summary = art.get("summary", "")
        source = art.get("source", "Unknown Source")

        emoji = _choose_emoji(title, summary)
        short_sum = _first_sentences(summary, max_sentences=3)

        lines.append(f"{emoji} {i}. {title}")
        if short_sum:
            # indent paragraphs
            wrapped = textwrap.fill(short_sum, width=78)
            lines.append(wrapped)
        lines.append(f"Source: {source}")
        lines.append("")

    # Read more links
    lines.append("🔗 Read More (Sources)")
    lines.append("")
    for art in selected:
        t = _clean_text(art.get("title", ""))
        l = art.get("link", "")
        if l:
            lines.append(f"{t} – {l}")

    body = "\n".join(lines)
    return subject, body


# ------------------ STEP 3: SEND EMAIL ------------------ #
def send_email(recipient_emails, subject: str, body: str):
    if not EMAIL_USER or not EMAIL_PASSWORD:
        raise ValueError("Email credentials (EMAIL_USER / EMAIL_PASSWORD) are not set in .env")

    if isinstance(recipient_emails, str):
        recipient_emails = [e.strip() for e in recipient_emails.split(",") if e.strip()]

    if not recipient_emails:
        raise ValueError("No recipient emails provided")

    msg = MIMEMultipart("alternative")
    msg["From"] = EMAIL_USER
    msg["To"] = ", ".join(recipient_emails)
    msg["Subject"] = subject

    # Create HTML version for better email formatting
    html_body = f"""
    <html>
        <head></head>
        <body style="font-family: Arial, sans-serif; line-height: 1.6; color: #333;">
            <div style="max-width: 600px; margin: 0 auto; padding: 20px;">
                <h2 style="color: #0066cc; border-bottom: 2px solid #0066cc; padding-bottom: 10px;">
                    AI Updates Digest
                </h2>
                <div style="margin: 20px 0; white-space: pre-wrap;">
{body}
                </div>
                <hr style="border: none; border-top: 1px solid #ddd; margin: 20px 0;">
                <p style="font-size: 12px; color: #666;">
                    This email was generated by AI Updates Email Sender
                </p>
            </div>
        </body>
    </html>
    """
    
    # Attach both plain text and HTML versions
    msg.attach(MIMEText(body, "plain"))
    msg.attach(MIMEText(html_body, "html"))

    with smtplib.SMTP(EMAIL_HOST, EMAIL_PORT) as server:
        server.starttls()
        server.login(EMAIL_USER, EMAIL_PASSWORD)
        server.send_message(msg)


# ------------------ SCHEDULER FUNCTIONS ------------------ #
def scheduled_digest(recipient_emails, max_items=15):
    """Automatically generate and send digest at scheduled time."""
    try:
        print(f"[{datetime.now()}] Starting scheduled digest...")
        articles = fetch_ai_news(max_items=max_items)
        
        if not articles:
            print("No AI articles found")
            return
        # Create formatted digest (local formatter)
        subject, digest = create_formatted_digest(articles)
        send_email(recipient_emails, subject, digest)
        print(f"[{datetime.now()}] Digest sent successfully!")
    except Exception as e:
        print(f"[{datetime.now()}] Error in scheduled digest: {e}")


def run_scheduler(recipient_emails, schedule_time, max_items=15):
    """Run scheduler in background thread."""
    try:
        import schedule as scheduler
    except ImportError:
        print("ERROR: 'schedule' module not installed. Install with: pip install schedule")
        return
    
    scheduler.every().day.at(schedule_time).do(scheduled_digest, recipient_emails, max_items)
    
    print(f"Scheduler started. Will send digest daily at {schedule_time}")
    while True:
        scheduler.run_pending()
        time.sleep(60)  # Check every minute


# ------------------ STREAMLIT APP ------------------ #
def main():
    st.set_page_config(page_title="AI Email Updates with Groq", page_icon="🤖", layout="centered")

    st.title("🤖 AI Updates Email Sender (Groq + Streamlit)")
    st.write(
        "This app fetches recent AI-related news from tech RSS feeds, summarizes them using **Groq LLM**, "
        "and sends a clean email digest to the address you provide."
    )

    st.markdown("### ✉️ Email Settings")
    recipient_input = st.text_input(
        "Recipient email(s)",
        placeholder="example@gmail.com or multiple, separated by commas",
        help="These emails will receive the AI updates digest."
    )

    show_preview = st.checkbox("Show digest preview before sending", value=True)

    st.markdown("### ⚙️ Options")
    max_items = st.slider("Maximum articles to fetch", min_value=5, max_value=30, value=15, step=5)
    use_groq = st.checkbox("Use Groq LLM to rewrite/expand the digest (optional)", value=False)

    st.markdown("### ⏰ Scheduler Settings (Optional)")
    enable_scheduler = st.checkbox("Enable daily automatic scheduling", value=False)
    
    if enable_scheduler:
        col1, col2 = st.columns(2)
        with col1:
            schedule_hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=12)
        with col2:
            schedule_minute = st.number_input("Minute (0-59)", min_value=0, max_value=59, value=0)
        
        schedule_time = f"{schedule_hour:02d}:{schedule_minute:02d}"
        
        if st.button("✅ Start Scheduler"):
            if not recipient_input.strip():
                st.error("Please enter recipient email(s) first!")
                return
            
            st.info(f"📅 Scheduler started! You will receive daily AI digest at {schedule_time}")
            st.warning("Keep this app running in the background for the scheduler to work.")
            
            # Start scheduler in background thread
            scheduler_thread = threading.Thread(
                target=run_scheduler,
                args=(recipient_input, schedule_time, max_items),
                daemon=True
            )
            scheduler_thread.start()
            st.success("Scheduler is now running!")

    st.markdown("---")

    if st.button("🚀 Generate & Send AI Digest"):
        if not recipient_input.strip():
            st.error("Please enter at least one recipient email.")
            return

        with st.spinner("Fetching AI news..."):
            try:
                articles = fetch_ai_news(max_items=max_items)
            except Exception as e:
                st.error(f"Error while fetching news: {e}")
                return

        st.success(f"Fetched {len(articles)} AI-related articles.")

        # Build digest: either use Groq (if requested and API key present) or the local formatter
        if use_groq and GROQ_API_KEY:
            with st.spinner("Summarizing with Groq..."):
                try:
                    digest = summarize_with_groq(articles)
                    today_str = datetime.now().strftime("%Y-%m-%d")
                    subject = f"AI Updates Digest – {today_str}"
                except Exception as e:
                    st.error(f"Error while summarizing with Groq: {e}")
                    return
        else:
            subject, digest = create_formatted_digest(articles)

        if show_preview:
            st.markdown("### 📝 Email Preview")
            st.code(digest, language="markdown")

        with st.spinner("Sending email..."):
            try:
                send_email(recipient_input, subject, digest)
            except Exception as e:
                st.error(f"Error while sending email: {e}")
                return

        st.success(f"✅ Email sent successfully to: {recipient_input}")


if __name__ == "__main__":
    main()
