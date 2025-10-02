import gradio as gr
from transformers import GPT2LMHeadModel, GPT2Tokenizer, pipeline
import torch

# =========================
# Load Fine-Tuned GPT-2 Model
# =========================
def load_model():
    model_path = "zaid002/finetunedmodel"   # Hugging Face repo path
    tokenizer = GPT2Tokenizer.from_pretrained(model_path)
    model = GPT2LMHeadModel.from_pretrained(model_path)
    device = 0 if torch.cuda.is_available() else -1
    text_gen = pipeline("text-generation", model=model, tokenizer=tokenizer, device=device)
    return tokenizer, text_gen

tokenizer, chatbot = load_model()

# =========================
# Translation Dictionary
# =========================
def translate_response(text, lang):
    translations = {
        "en": text,
        "hi": "यह रहा आपका उत्तर हिंदी में: " + text,
        "ta": "உங்கள் பதில் தமிழில்: " + text,
        "te": "మీ సమాధానం తెలుగులో: " + text,
        "fr": "Votre réponse en français : " + text,
        "es": "Su respuesta en español: " + text
    }
    return translations.get(lang, text)

# =========================
# Chat Function
# =========================
def multilingual_chat(user_input, history, lang="en"):
    # Add prefix for better response format
    input_text = f"Question: {user_input}\nAnswer:"

    raw_response = chatbot(
        input_text,
        max_length=100,
        num_return_sequences=1,
        pad_token_id=tokenizer.eos_token_id,
        do_sample=True,
        top_k=50,
        top_p=0.9,
        temperature=0.7
    )
    
    # Remove the question part from output
    model_output = raw_response[0]["generated_text"].replace(input_text, "").strip()

    # Limit length to avoid repetition
    if len(model_output) > 300:
        model_output = model_output[:300]

    # Translate if needed
    response = translate_response(model_output, lang)

    # Save conversation
    history.append((user_input, response))
    return history, history

# =========================
# Gradio UI
# =========================
with gr.Blocks(theme=gr.themes.Soft()) as demo:
    gr.HTML(
        """
        <div style="font-family: Calibri; text-align:center;">
            <h1>🤖 Multilingual LLM Chatbot</h1>
            <p>Ask me anything! Answers are generated from your fine-tuned GPT-2 model.</p>
        </div>
        """
    )

    chatbot_ui = gr.Chatbot(height=450)
    with gr.Row():
        msg = gr.Textbox(placeholder="💬 Type your question...", label="Your Message")
        lang = gr.Dropdown(["en", "hi", "ta", "te", "fr", "es"], label="Language", value="en")

    clear = gr.Button("🗑️ Clear Chat")
    state = gr.State([])

    msg.submit(multilingual_chat, [msg, state, lang], [chatbot_ui, state])
    clear.click(lambda: ([], []), None, [chatbot_ui, state], queue=False)

# =========================
# Launch
# =========================
if __name__ == "__main__":
    demo.launch()
