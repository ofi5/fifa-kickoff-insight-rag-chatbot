
# FIFA World Cup ChatBot

A Streamlit-based RAG chatbot that answers questions about the FIFA World Cup. It builds a knowledge base from local PDFs **and** a fixed set of crawled FIFA/World Cup web pages, embeds them locally with a HuggingFace sentence-transformer (no embedding API needed), stores them in FAISS, and answers with a Groq LLM.

## Project Structure

```
QaChatBot/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
├── data/              # Data directory for PDF files
│   └── sampledata.pdf # FIFA World Cup 26 fact sheet (sample source)
├── .env               # Environment variables (create this file)
├── .gitignore         # Git ignore rules
└── README.md          # This file
```

## Setup Instructions

1. **Install Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

2. **Set up API Keys:**
   
   **For Local Development:**
   Only a Groq key is required (embeddings run locally). Create a
   `.streamlit/secrets.toml` (or `.env`) with:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   ```
   
   **For Website Deployment (Streamlit Cloud):**
   Use Streamlit secrets. In your Streamlit Cloud dashboard:
   - Go to your app settings
   - Navigate to "Secrets"
   - Add your API keys in this format:
   ```toml
   GROQ_API_KEY = "your_groq_api_key_here"
   GOOGLE_API_KEY = "your_google_api_key_here"
   ```

3. **Add Your PDF Files:**
   Place your PDF files in the `data/` directory. The app will look for `sampledata.pdf` by default.

4. **Run the Application:**
   ```bash
   streamlit run app.py
   ```

## Features

- PDF document processing (loads every PDF in `data/`)
- Web crawling of a fixed list of FIFA / World Cup sources (Wikipedia)
- Local vector embeddings via HuggingFace `all-MiniLM-L6-v2` (no API key, no rate limit)
- FAISS vector store persisted to `faiss_index/` (built once, reused on later runs)
- Question answering using a Groq LLM (`llama-3.3-70b-versatile`)
- Source attribution for each answer
- Streamlit web interface with source toggles and a build/rebuild button

## Usage

1. Click "Create Vector Store" to initialize the embeddings and load the PDF
2. Enter your question in the text input
3. Get answers based on the content of the FIFA World Cup fact sheet
4. View document similarity search results in the expandable section

## Deployment

When deploying to your website:
- Ensure the `data/` directory contains your PDF files
- Set up Streamlit secrets with your API keys (GROQ_API_KEY and GOOGLE_API_KEY)
- The virtual environment (`venv/`) should not be included in deployment
- For Streamlit Cloud: Add your secrets in the app settings
