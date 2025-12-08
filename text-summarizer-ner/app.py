"""
Summarizer & NER Tool
---------------------
A Gradio-based web interface that uses Hugging Face Transformers to:
1. Summarize long text using DistilBART.
2. Perform Named Entity Recognition (NER) on the summary using BERT.
"""

import os
import logging
import gradio as gr
from typing import List, Tuple, Optional
from dataclasses import dataclass

# Third-party imports
from transformers import pipeline

# --- Configuration ---
@dataclass
class AppConfig:
    """Global configuration constants."""
    SUM_MODEL: str = "sshleifer/distilbart-cnn-12-6"
    NER_MODEL: str = "dslim/bert-base-NER"
    TITLE: str = "Text Summarizer & Named Entity Recognition"
    DESC: str = """
    ### Text Analysis
    **1. Summarize:** Reduces text into concise paragraphs using `DistilBART`. \n
    **2. Analyze:** Extracts entities (Persons, Organizations, Locations) using `BERT-NER`.
    """
    # Standard example text for users to test immediately
    SAMPLE_TEXT: str = (
        "Apollo 11 was the American spaceflight that first landed humans on the Moon. "
        "Commander Neil Armstrong and lunar module pilot Buzz Aldrin landed the Apollo Lunar Module Eagle on July 20, 1969. "
        "Armstrong became the first person to step onto the lunar surface six hours and 39 minutes later on July 21 at 02:56 UTC; "
        "Aldrin joined him 19 minutes later. They spent about two and a quarter hours together outside the spacecraft, "
        "and they collected 47.5 pounds (21.5 kg) of lunar material to bring back to Earth. "
        "Michael Collins flew the Command Module Columbia alone in lunar orbit while they were on the Moon's surface."
    )

# --- Logging Setup ---
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Suppress Hugging Face symlink warnings on Windows
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"


class TextProcessor:
    """
    Encapsulates model loading and inference logic to keep the UI clean.
    """
    def __init__(self):
        self.summarizer = None
        self.ner_pipeline = None
        self._load_models()

    def _load_models(self):
        """Initializes HF pipelines with error handling."""
        try:
            logger.info(f"Loading Summarizer: {AppConfig.SUM_MODEL}...")
            self.summarizer = pipeline("summarization", model=AppConfig.SUM_MODEL)

            logger.info(f"Loading NER: {AppConfig.NER_MODEL}...")
            # aggregation_strategy='simple' merges sub-tokens (e.g., "New", "York" -> "New York")
            self.ner_pipeline = pipeline("ner", model=AppConfig.NER_MODEL, aggregation_strategy="simple")
            
            logger.info("✅ Models loaded successfully.")
        except Exception as e:
            logger.error(f"❌ Model loading failed: {e}")

    def process(self, text: str, max_len: int, min_len: int) -> Tuple[str, List]:
        """
        Runs the summarization and NER inference pipeline.
        
        Returns:
            Tuple(summary_text, formatted_ner_list)
        """
        if not self.summarizer or not self.ner_pipeline:
            return "⚠️ Error: Models not initialized. See logs.", []

        if not text or len(text.strip()) < 50:
            return "⚠️ Text too short. Please provide at least 50 characters.", []

        try:
            # 1. Summarize
            summary_res = self.summarizer(
                text, 
                max_length=int(max_len), 
                min_length=int(min_len), 
                do_sample=False
            )
            summary_text = summary_res[0]['summary_text']

            # 2. NER Analysis
            ner_res = self.ner_pipeline(summary_text)

            # 3. Format for Gradio HighlightedText
            # We must stitch the text back together to show non-entities + entities
            formatted_output = []
            cursor = 0
            
            for entity in ner_res:
                # Text before entity
                if entity['start'] > cursor:
                    formatted_output.append((summary_text[cursor:entity['start']], None))
                
                # The Entity
                formatted_output.append((summary_text[entity['start']:entity['end']], entity['entity_group']))
                cursor = entity['end']
            
            # Remaining text
            if cursor < len(summary_text):
                formatted_output.append((summary_text[cursor:], None))

            return summary_text, formatted_output

        except Exception as e:
            logger.error(f"Inference Error: {e}")
            return f"Error during processing: {e}", []


# --- JavaScript Helpers ---
# Used for the manual copy button
JS_COPY = """
(text) => {
    navigator.clipboard.writeText(text);
    return text;
}
"""

# --- Main UI Construction ---
def build_interface():
    processor = TextProcessor()
    with gr.Blocks(title=AppConfig.TITLE) as app:
        
        # Header
        gr.Markdown(f"# {AppConfig.TITLE}")
        gr.Markdown(AppConfig.DESC)

        with gr.Row():
            
            # --- Left Column: Input & Controls ---
            with gr.Column(scale=1):
                input_box = gr.Textbox(
                    label="Source Text", 
                    lines=10, 
                    placeholder="Paste article here..."
                )
                
                with gr.Accordion("⚙️ Settings", open=False):
                    slider_max = gr.Slider(50, 300, value=130, step=10, label="Max Length")
                    slider_min = gr.Slider(10, 100, value=30, step=5, label="Min Length")

                with gr.Row():
                    btn_clear = gr.Button("🗑️ Clear", variant="secondary")
                    btn_submit = gr.Button("✨ Summarize & Analyze", variant="primary")

            # --- Right Column: Results ---
            with gr.Column(scale=1):
                # Hidden/Plain text field for the copy function
                out_summary = gr.Textbox(label="Plain Summary", lines=4, interactive=False)
                
                # Visual field for NER
                out_ner = gr.HighlightedText(
                    label="Entity Visualization",
                    combine_adjacent=True,
                    show_legend=True,
                    color_map={"PER": "blue", "ORG": "green", "LOC": "orange", "MISC": "gray"}
                )
                
                btn_copy = gr.Button("📋 Copy Summary to Clipboard")

        # --- Footer & Examples ---
        gr.Examples([AppConfig.SAMPLE_TEXT], inputs=[input_box], label="Try an Example")
        
        gr.Markdown("---")
        gr.Markdown(f"**Engine:** `{AppConfig.SUM_MODEL}` + `{AppConfig.NER_MODEL}`")

        # --- Interactions ---
        btn_submit.click(
            processor.process, 
            inputs=[input_box, slider_max, slider_min], 
            outputs=[out_summary, out_ner]
        )

        btn_clear.click(
            lambda: ("", "", []), 
            outputs=[input_box, out_summary, out_ner]
        )
        
        # JavaScript Copy Trigger
        btn_copy.click(None, inputs=[out_summary], js=JS_COPY)

    return app

if __name__ == "__main__":
    demo = build_interface()
    demo.launch()