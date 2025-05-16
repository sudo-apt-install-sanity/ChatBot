# Install dependencies (needed only in Colab or if not installed locally)
#!pip install gradio pandas scikit-learn matplotlib --quiet

# --- Imports ---
import pandas as pd
import re
import gradio as gr
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import make_pipeline

# --- Dataset ---
data = {
    "User": [
        "hi", "hello", "what is your name", "where are you located",
        "what are your hours", "i need support", "yes", "no"
    ],
    "Bot": [
        "Hello! How can I assist you today?",
        "Hi there! How can I help?",
        "I'm SupportBot, your virtual assistant.",
        "We are an online service, available globally!",
        "Our support team is available 24/7.",
        "Sure! How can I assist you today?",
        "Great! Please tell me how I can help.",
        "Alright. Let me know if you need anything."
    ]
}
df = pd.DataFrame(data)

# --- Clean text ---
def clean_text(text):
    return re.sub(r"[^a-zA-Z\s]", "", text).lower().strip()

df["Cleaned_User"] = df["User"].apply(clean_text)

# --- Train model ---
model = make_pipeline(CountVectorizer(), MultinomialNB())
model.fit(df["Cleaned_User"], df["Bot"])

# --- Gradio function ---
def chatbot(user_input, history=[]):
    cleaned = clean_text(user_input)
    response = model.predict([cleaned])[0]
    history.append((user_input, response))
    return history, history

# --- Gradio UI ---
with gr.Blocks() as demo:
    gr.Markdown("## 🤖 Intelligent Chatbot (SupportBot)")
    chatbot_ui = gr.Chatbot()
    msg = gr.Textbox(label="Type your message here...")
    clear = gr.Button("Clear")

    state = gr.State([])

    msg.submit(chatbot, [msg, state], [chatbot_ui, state])
    clear.click(lambda: ([], []), None, [chatbot_ui, state])

# --- Launch ---
demo.launch(debug=False)
