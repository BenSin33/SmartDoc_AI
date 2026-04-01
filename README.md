# SmartDoc AI
Intelligent Document Q&A SystemSmartDoc AI is an intelligent document questioning and answering system built using Retrieval-Augmented Generation (RAG). This system allows users to upload PDF files and interact directly with the content through a local Large Language Model (LLM), ensuring absolute data privacy and security.  
# 🚀 Key FeaturesMultilingual PDF Processing: 
Accurately extracts text from PDFs, supporting Vietnamese, English, and over 50 other languages.  
Real-time Q&A: Utilizes the Qwen2.5:7b model optimized for high-quality Vietnamese and multilingual support to provide concise answers.  
Semantic Search: Integrates the FAISS vector database and MPNet embedding models to find information based on context and meaning rather than just keywords.  
User-Friendly Web Interface: Built on the Streamlit framework, featuring a drag-and-drop file uploader and intuitive processing progress displays.  
Local & Offline Operation: Runs entirely on a personal computer via Ollama, requiring no internet connection after the initial model download and incurring no API costs.  
# 🛠 Technology Stack
The system is designed with a multi-layer architecture:
Presentation Layer: Streamlit (v1.41.1).
Application Layer: LangChain (v0.3.16).
Data Layer: FAISS (v1.9.0) for vector storage and PDFPlumber for precise document loading.
Model Layer: Ollama runtime powering the Qwen2.5:7b model.
# 📋 System Requirements
OS: Windows, macOS, or Linux.
Language: Python 3.13+.
Software: Ollama runtime.
Package Manager: pip.
# 🔧 Installation and Usage
1. Environment SetupBash# Clone the repository

git clone [repository-url]

cd Project-LLMs-Rag-Agent

--Create and activate a virtual environment

python -m venv venv

--For Windows:

venv\Scripts\activate

--For Linux/Mac:

source venv/bin/activate

--Install required dependencies:

pip install -r requirements.txt

2. Model Installation
   (Ollama)Download and install Ollama from ollama.ai, then pull the required model:
   ollama pull qwen2.5:7b
3. Running the Application:
   streamlit run app.py
Once started, access the interface at http://localhost:8501 in your web browser.

# 📂 Project Structure
app.py: The main application entry point.
requirements.txt: List of Python dependencies.
data/: Directory containing sample PDF documents.
documentation/: Project reports and LaTeX documentation.
venv/: Python virtual environment.

# 💡 Usage TipsSpecific Questions: 
For the most accurate results, ask specific and clear questions using keywords found in the document.
Language Detection: The system automatically detects the input language and responds accordingly in either Vietnamese or English.
Troubleshooting: If the system fails to respond, ensure that the Ollama service is running on your machine.This project was developed for the Open Source Software Development (OSSD) course at the Faculty of Information Technology - Saigon University (2026).
