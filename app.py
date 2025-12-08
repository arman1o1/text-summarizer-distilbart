"""
Text Summarizer using Hugging Face Transformers and Gradio.
"""

import os
import logging
import gradio as gr
from transformers import pipeline

# --- Configuration & Setup ---
# Configure logging to show timestamps and levels
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Disable Hugging Face symlink warnings for Windows users
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"

# Constants
MODEL_NAME = "sshleifer/distilbart-cnn-12-6"
APP_TITLE = "Text Summarizer using DistilBART"
APP_DESCRIPTION = """
### Summarize text using the distilbart-cnn-12-6 model.
"""

SAMPLE_TEXT = """
Apollo 11 was the American spaceflight that first landed humans on the Moon. Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC; Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, and they collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface. Armstrong and Aldrin spent 21 hours, 36 minutes on the lunar surface before lifting off to rejoin Columbia.
"""

# --- Model Initialization ---
try:
    logger.info(f"Loading model: {MODEL_NAME}...")
    summarizer = pipeline("summarization", model=MODEL_NAME)
    logger.info("Model loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load model: {e}")
    summarizer = None


# --- Core Logic ---
def summarize_text(text: str, max_len: int, min_len: int) -> str:
    """
    Generates a summary for the provided text using the loaded model.
    """
    if summarizer is None:
        return "Error: Model not loaded correctly. Check logs."

    if not text or not text.strip():
        return "⚠️ Please enter some text to summarize."

    if len(text) < 50:
        return "⚠️ Input text is too short. Please provide a longer paragraph."

    try:
        logger.info("Processing summary request...")
        summary_list = summarizer(
            text,
            max_length=int(max_len),
            min_length=int(min_len),
            do_sample=False  # Deterministic output
        )
        return summary_list[0]['summary_text']
    except Exception as e:
        logger.error(f"Summarization error: {e}")
        return f"An error occurred: {str(e)}"


def clear_fields():
    """Resets the input and output text boxes."""
    return "", ""


# --- Client-Side JavaScript ---
COPY_SCRIPT = """
(text) => {
    navigator.clipboard.writeText(text);
    return text;
}
"""

# --- User Interface ---
def create_interface():
    # Removed 'theme' argument to ensure compatibility with your version
    with gr.Blocks(title=APP_TITLE) as demo:
        
        # Header
        gr.Markdown(f"# {APP_TITLE}")
        gr.Markdown(APP_DESCRIPTION)

        # Main Split Layout
        with gr.Row():
            
            # LEFT COLUMN: Inputs
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    label="Input Text",
                    lines=12,
                    placeholder="Paste your article or text here..."
                )

                # Advanced Settings Accordion
                with gr.Accordion("⚙️ Model Settings", open=False):
                    max_len_slider = gr.Slider(
                        minimum=50, maximum=300, value=150, step=10, 
                        label="Max Length"
                    )
                    min_len_slider = gr.Slider(
                        minimum=10, maximum=100, value=30, step=5, 
                        label="Min Length"
                    )

                # Action Buttons
                with gr.Row():
                    clear_btn = gr.Button("🗑️ Clear", variant="secondary")
                    submit_btn = gr.Button("Summarize", variant="primary")

            # RIGHT COLUMN: Outputs
            with gr.Column(scale=1):
                # Removed 'show_copy_button' to ensure compatibility
                output_box = gr.Textbox(
                    label="Summary",
                    lines=12,
                    interactive=False
                )
                
                # Copy Button
                copy_btn = gr.Button("📋 Copy to Clipboard")

        # Examples Section
        gr.Examples(
            examples=[[SAMPLE_TEXT]],
            inputs=[input_box],
            label="Try an Example"
        )
        
        # Footer
        gr.Markdown("---")
        gr.Markdown(f"**Model:** `{MODEL_NAME}` | **Device:** CPU Mode")

        # Event Wiring
        submit_btn.click(
            fn=summarize_text,
            inputs=[input_box, max_len_slider, min_len_slider],
            outputs=[output_box]
        )

        clear_btn.click(
            fn=clear_fields,
            inputs=None,
            outputs=[input_box, output_box]
        )

        copy_btn.click(
            fn=None,
            inputs=[output_box],
            outputs=None,
            js=COPY_SCRIPT
        )

    return demo


if __name__ == "__main__":
    app = create_interface()
    app.launch()