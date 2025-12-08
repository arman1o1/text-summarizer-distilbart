# Text Summarizer with DistilBART

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?style=for-the-badge&logo=python&logoColor=white)
![Hugging Face](https://img.shields.io/badge/Hugging%20Face-Transformers-yellow?style=for-the-badge&logo=huggingface&logoColor=black)
![Gradio](https://img.shields.io/badge/Gradio-UI-orange?style=for-the-badge&logo=gradio&logoColor=white)

A fast, privacy-focused AI application that automatically generates concise summaries from long articles or text documents. Built using Python, the **DistilBART** model, and **Gradio** for a clean, split-view interface.

## App Demo Screenshot
![App Demo Interface](demo.png)

## 🚀 Features

*   **100% Offline:** Runs entirely locally on your machine after the initial model download. Your data never leaves your computer.
*   **No API Keys:** Does not require a Hugging Face token or OpenAI key. Completely free to run.
*   **Split-View Interface:** Compare the original text and the summary side-by-side.
*   **Customizable:** Adjust the **Minimum** and **Maximum** length of the summary using simple sliders.
*   **Deterministic Output:** Uses optimized search settings to ensure consistent, high-quality results.

##  Installation

### 1. Clone the Repository
```bash
git clone https://github.com/YOUR_USERNAME/offline-text-summarizer.git
cd offline-text-summarizer
```

### 2. Install Dependencies
It is recommended to use a virtual environment.
```bash
pip install -r requirements.txt
```

##  Usage

Run the application using Python:

```bash
python app.py
```

Wait for the model to load (approx. 1-2 minutes on the first run). Once ready, the terminal will show a local URL:

```text
Running on local URL:  http://127.0.0.1:7860
```

Open that link in your browser to use the app.

##  Technical Details

*   **Model:** [sshleifer/distilbart-cnn-12-6](https://huggingface.co/sshleifer/distilbart-cnn-12-6)
*   **Task:** Abstractive Summarization
*   **Framework:** PyTorch & Transformers
*   **Logic:** The app uses a Hugging Face `pipeline`. It downloads the model weights (~1.2GB) to your local cache. The summarization logic is set to `do_sample=False` to ensure mathematically optimal summaries rather than random text generation.

## 📄 License

The code in this repository is licensed under the [MIT License](LICENSE).
The model used (`distilbart-cnn-12-6`) is licensed under **Apache 2.0**.
