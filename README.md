📑 SmartDoc AI: Intelligent Document Q&A System
SmartDoc AI là hệ thống hỏi đáp tài liệu thông minh dựa trên kỹ thuật Retrieval-Augmented Generation (RAG). Hệ thống cho phép người dùng tương tác trực tiếp với các tệp PDF thông qua mô hình ngôn ngữ lớn (LLM) chạy cục bộ, đảm bảo tính riêng tư tuyệt đối và bảo mật dữ liệu.

🚀 Tính năng nổi bật
Xử lý tài liệu đa ngôn ngữ: Trích xuất văn bản từ PDF chính xác, hỗ trợ tiếng Việt, tiếng Anh và hơn 50 ngôn ngữ khác.

Hỏi đáp thời gian thực: Sử dụng model Qwen2.5:7b tối ưu cho tiếng Việt, cung cấp câu trả lời mạch lạc, đúng trọng tâm.

Tìm kiếm ngữ nghĩa (Semantic Search): Kết hợp cơ sở dữ liệu vector FAISS và mô hình embedding MPNet để hiểu ngữ cảnh thay vì chỉ khớp từ khóa đơn thuần.

Hoạt động Offline hoàn toàn: Chạy cục bộ thông qua Ollama, không cần kết nối internet sau khi tải model, không phát sinh chi phí API.

Giao diện thân thiện: Phát triển trên nền tảng Streamlit với tính năng kéo thả tệp và hiển thị tiến trình xử lý trực quan.

🛠 Kiến trúc hệ thống
Dự án được thiết kế theo kiến trúc đa lớp đảm bảo tính module hóa:

Presentation Layer: Streamlit (v1.41.1) - Xử lý giao diện người dùng.

Application Layer: LangChain (v0.3.16) - Điều phối luồng dữ liệu RAG.

Data Layer: * PDFPlumber: Trích xuất nội dung văn bản chất lượng cao.

FAISS (v1.9.0): Lưu trữ và truy vấn vector.

Model Layer: Ollama runtime vận hành model Qwen2.5:7b.

📋 Yêu cầu hệ thống
Hệ điều hành: Windows, macOS, hoặc Linux.

Ngôn ngữ: Python 3.8 trở lên.

Phần mềm hỗ trợ: Ollama runtime.

RAM đề nghị: Tối thiểu 8GB (Khuyến khích 16GB để model chạy mượt mà).

🔧 Cài đặt & Sử dụng
1. Thiết lập môi trường
# Clone repository
git clone https://github.com/BenSin33/SmartDoc_AI.git
cd SmartDoc_AI

# Tạo và kích hoạt môi trường ảo
python -m venv venv

# Windows:
venv\Scripts\activate
# Linux/Mac:
source venv/bin/activate

# Cài đặt các thư viện cần thiết
pip install -r requirements.txt

2. Cài đặt Model (Ollama)
Tải và cài đặt Ollama tại ollama.ai, sau đó thực hiện lệnh:
ollama pull qwen2.5:7b

3. Chạy ứng dụng
streamlit run app.py

📂 Cấu trúc dự án
├── app.py                # Điểm khởi chạy ứng dụng chính
├── requirements.txt      # Danh sách các thư viện phụ thuộc
├── data/                 # Thư mục chứa các tài liệu PDF mẫu
├── documentation/        # Báo cáo dự án và tài liệu LaTeX
└── venv/                 # Môi trường ảo Python

💡 Lưu ý khi sử dụng
Đặt câu hỏi cụ thể: Để có kết quả tốt nhất, hãy đưa ra câu hỏi rõ ràng dựa trên nội dung có trong tài liệu đã upload.

Tự động nhận diện ngôn ngữ: Hệ thống sẽ tự động phản hồi bằng ngôn ngữ tương ứng với câu hỏi của bạn (Tiếng Việt hoặc Tiếng Anh).

Xử lý sự cố: Nếu hệ thống không phản hồi, hãy kiểm tra chắc chắn rằng dịch vụ Ollama đang chạy ngầm trên máy tính của bạn.

MIT License

Copyright (c) 2026 Nhóm 15 môn học OSSD Trường Đại học Sài Gòn

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
