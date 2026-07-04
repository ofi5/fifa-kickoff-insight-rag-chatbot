
# FIFA World Cup ChatBot

A Streamlit-based RAG chatbot that answers questions about the FIFA World Cup using Google Generative AI embeddings and a Groq LLM.

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
   Create a `.env` file in the root directory with your API keys:
   ```
   GROQ_API_KEY=your_groq_api_key_here
   GOOGLE_API_KEY=your_google_api_key_here
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

- PDF document processing and text extraction
- Vector embeddings using Google Generative AI
- Question answering using Groq LLM
- Document similarity search
- Streamlit web interface

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
