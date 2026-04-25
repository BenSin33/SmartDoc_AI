📑 SmartDoc AI — Intelligent Document Q&A System

🚀 Hệ thống hỏi đáp tài liệu thông minh sử dụng Retrieval-Augmented Generation (RAG)
🔒 Chạy hoàn toàn cục bộ — bảo mật dữ liệu tuyệt đối

📌 Giới thiệu

SmartDoc AI là một hệ thống cho phép người dùng đặt câu hỏi trực tiếp trên tài liệu PDF bằng cách kết hợp giữa:

🔍 Semantic Search (tìm kiếm ngữ nghĩa)
🤖 Large Language Models (LLM)
🧠 Vector Database

Khác với chatbot thông thường, SmartDoc AI chỉ trả lời dựa trên nội dung tài liệu, đảm bảo độ chính xác và tránh “ảo giác AI”.

✨ Tính năng nổi bật
🌐 Đa ngôn ngữ
Hỗ trợ tiếng Việt, tiếng Anh và hơn 50+ ngôn ngữ
Trích xuất nội dung PDF chính xác cao
⚡ Hỏi đáp thời gian thực
Sử dụng model Qwen2.5:7B
Tối ưu cho tiếng Việt, trả lời rõ ràng, đúng trọng tâm
🧠 Semantic Search
Sử dụng:
FAISS (Vector Database)
MPNet Embedding
Hiểu ngữ cảnh thay vì chỉ so khớp từ khóa
🔒 Hoạt động Offline 100%
Chạy local qua Ollama
Không cần API → không tốn chi phí
Dữ liệu không rời khỏi máy
🖥 Giao diện thân thiện
Xây dựng bằng Streamlit
Kéo & thả file PDF
Hiển thị tiến trình xử lý trực quan
🏗 Kiến trúc hệ thống

Hệ thống được thiết kế theo kiến trúc nhiều lớp:

┌──────────────────────────┐
│ Presentation Layer       │ → Streamlit UI
├──────────────────────────┤
│ Application Layer        │ → LangChain (RAG Pipeline)
├──────────────────────────┤
│ Data Layer               │ → PDFPlumber + FAISS
├──────────────────────────┤
│ Model Layer              │ → Ollama (Qwen2.5:7B)
└──────────────────────────┘
🛠 Công nghệ sử dụng
Thành phần	Công nghệ
UI	Streamlit
Backend	LangChain
LLM	Qwen2.5:7B (Ollama)
Vector DB	FAISS
Embedding	MPNet
PDF Processing	PDFPlumber
📋 Yêu cầu hệ thống
💻 OS: Windows / macOS / Linux
🐍 Python: >= 3.8
🧠 RAM:
Tối thiểu: 8GB
Khuyến nghị: 16GB
⚙️ Cài đặt Ollama
⚙️ Cài đặt & chạy dự án
1️⃣ Clone repository
git clone https://github.com/BenSin33/SmartDoc_AI.git
cd SmartDoc_AI
2️⃣ Tạo môi trường ảo
python -m venv venv

Kích hoạt:

Windows:
venv\Scripts\activate
Linux / macOS:
source venv/bin/activate
3️⃣ Cài dependencies
pip install -r requirements.txt
4️⃣ Cài đặt model (Ollama)

Tải Ollama tại: https://ollama.ai

Sau đó chạy:

ollama pull qwen2.5:7b
5️⃣ Chạy ứng dụng
streamlit run app.py
📂 Cấu trúc dự án
SmartDoc_AI/
│
├── app.py                # Entry point
├── requirements.txt      # Dependencies
├── data/                 # Sample PDF files
├── documentation/        # Báo cáo & tài liệu LaTeX
└── venv/                 # Virtual environment
💡 Hướng dẫn sử dụng
📌 Đặt câu hỏi cụ thể → kết quả chính xác hơn
🌍 Tự động nhận diện ngôn ngữ (VI / EN)
⚠️ Nếu không phản hồi:
Kiểm tra Ollama đã chạy chưa
🚀 Định hướng phát triển
 Hỗ trợ nhiều định dạng (DOCX, TXT, HTML)
 Multi-user system
 Cloud deployment
 Fine-tuned model riêng
📄 License

MIT License

Copyright (c) 2026 Nhóm 15 - OSSD - Đại học Sài Gòn

Permission is hereby granted, free of charge...
