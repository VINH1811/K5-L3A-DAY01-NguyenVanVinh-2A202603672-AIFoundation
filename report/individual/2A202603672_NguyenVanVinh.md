# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

> Đổi tên file này thành `MSSV_HoTen.md` của bạn, ví dụ `2A20260000_NguyenVanAn.md`, rồi điền từng mục. Xoá các dòng hướng dẫn in nghiêng khi xong.

- **MSSV: 2A202603672**
- **Họ và tên: Nguyễn Văn Vinh**
- **SEED** (in ở ô đầu notebook): 3608

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

## 2. Kết quả chính

                                 precision  recall  F1 (test)  F1 (SUB 40)  \
1. BoW + LogReg (tự code)            0.962   0.789      0.867        0.824   
2. BoW + MLP (deep learning)         0.968   0.958      0.963        1.000   
3a. Encoder (MiniLM) + kNN           0.835   0.958      0.892        0.950   
3b. Encoder (MiniLM) + LogReg        0.938   0.947      0.942        0.974   
3c. Encoder (MiniLM) + MLP head      0.939   0.979      0.959        1.000   
4a. LLM nhỏ zero-shot                0.870   1.000      0.930        0.930   
4b. LLM lớn zero-shot                0.870   1.000      0.930        0.930   
4c. LLM nhỏ few-shot                 0.870   1.000      0.930        0.930   

                                 train (s)  tin/s (cả lô)  p50 ms/tin  \
1. BoW + LogReg (tự code)            4.988      25682.175       0.061   
2. BoW + MLP (deep learning)         6.733      45419.270       0.149   
3a. Encoder (MiniLM) + kNN          98.479         22.755      62.687   
3b. Encoder (MiniLM) + LogReg      100.114         27.007      29.460   
3c. Encoder (MiniLM) + MLP head     99.420         26.895      28.696   
4a. LLM nhỏ zero-shot                0.000          1.102     399.932   
4b. LLM lớn zero-shot                0.000          1.208     373.992   
4c. LLM nhỏ few-shot                 0.000          1.202     344.434   

                                 p95 ms/tin  USD / 1k tin  
1. BoW + LogReg (tự code)             0.083           0.0  
2. BoW + MLP (deep learning)          0.188           0.0  
3a. Encoder (MiniLM) + kNN          185.259           0.0  
3b. Encoder (MiniLM) + LogReg        37.632           0.0  
3c. Encoder (MiniLM) + MLP head      37.336           0.0  
4a. LLM nhỏ zero-shot              2474.138           0.0  
4b. LLM lớn zero-shot              2394.565           0.0  
4c. LLM nhỏ few-shot               2458.092           0.0  
• Hàng LLM chỉ đo trên SUB (40 tin) → so sánh chất lượng công bằng bằng cột 'F1 (SUB 40)'.
• SUB chỉ 40 tin: 1 tin sai đổi F1 vài điểm. Chênh < ~0.05 F1 coi như HOÀ — đừng xếp hạng theo số lẻ.
• Inference đo TỪ VĂN BẢN THÔ → NHÃN (gồm cả vectorize / encode). p50/p95 = gọi từng tin một, như một request thật.
• train (s) của 3a/3b/3c = thời gian encoder MiniLM encode tập train (+ train đầu LogReg / MLP); LLM = 0 vì không train. Số đo phụ thuộc CPU/GPU của Colab.


## 3. Dự đoán và thực tế

| Dự đoán | Em dự đoán | Kết quả thực tế | Vì sao lệch? |
|---|---|---|---|
| `hard_eval` | BoW + MLP đoán đúng khoảng 65% trên 50 tin khó. | Đúng 26/50 tin (52%), thấp hơn dự đoán 13 điểm phần trăm. | Em đánh giá quá cao khả năng nhận diện spam viết như lời trò chuyện. Mô hình bỏ sót cả 10 tin thuộc nhóm này. |
| `llm_models` | Groq 120B sẽ hơn 20B khoảng 0,02 F1 và nhỉnh hơn MiniLM + LogReg. | Hai LLM cùng đạt F1 = 0,930; MiniLM + LogReg đạt 0,974 trên cùng 40 tin. | Em cho rằng model lớn hơn sẽ phân loại tốt hơn. Thực tế, cả hai LLM đều bắt đủ 20 tin spam nhưng cùng chặn nhầm 3 tin thường. |
| `ml_f1` | BoW + LogReg sẽ đạt F1 khoảng 0,80, precision cao hơn recall. | F1 đạt 0,867, cao hơn dự đoán 0,067. Precision = 0,962 và recall = 0,789. | Em dự đoán đúng xu hướng nhưng đánh giá thấp khả năng của mô hình: trong 78 tin bị chặn, có tới 75 tin thực sự là spam. |


## 4. Phân tích 50 case khó

**BoW + MLP giảm mạnh nhất:** F1 từ **1,000** trên 40 tin SUB xuống **0,368** trên 50 tin khó, giảm **0,632**. Nhóm khó nhất là *spam giọng tự nhiên*: mô hình đoán sai cả **10/10 tin**.

- **Case #11 – spam viết như lời nhắn cá nhân:** Tin mở đầu bằng “hi there, sorry to bother you, my friend…”. BoW + MLP đoán `ham`. Một cách giải thích là những từ chào hỏi quen thuộc khiến tin giống hội thoại đời thường, trong khi BoW khó thể hiện ý đồ của cả câu.
- **Case #47 – tin thường kể lại một vụ lừa đảo:** Người gửi kể mình nhận được tin báo trúng £1.000 rồi hỏi còn ai mắc lừa không. Groq 20B zero-shot nhận đúng là `ham`, còn bản dùng prompt phòng thủ chặn nhầm. Có thể các chi tiết “trúng tiền” và “gọi số 0906” lấn át ngữ cảnh người gửi đang cảnh báo về spam.

## 5. Thử thách cá nhân

Với **SEED = 3608**, em nhận **đề số 3**: so sánh MLP có 4 và 512 nơ-ron ẩn. Em giữ nguyên dữ liệu BoW, seed và số epoch (30), chỉ thay kích thước lớp ẩn. Em đo riêng thời gian huấn luyện và tính F1 trên toàn bộ tập test.

| Nơ-ron ẩn | Số tham số | F1 test | Thời gian train |
|---:|---:|---:|---:|
| 4 | 8.009 | 0,963 | 5,769 giây |
| 512 | 1.025.025 | 0,963 | 57,432 giây |

## 6. Phương án cho sản phẩm thật

Em chọn **BoW + MLP làm bộ lọc chính**. Trong phép đo của lab, mô hình đạt **F1 test = 0,963**, **p95 = 0,188 ms/tin** và xử lý khoảng **45.419 tin/giây theo lô**. Các số đo này đáp ứng mục tiêu p95 dưới 50 ms và vượt lưu lượng trung bình khoảng 116 tin/giây của bài toán 10 triệu SMS/ngày.

Đánh đổi lớn nhất là BoW + MLP chỉ đạt **F1 = 0,368** trên bộ tin khó. Em đề xuất chuyển các tin có xác suất spam từ **0,3 đến 0,7** sang Groq 20B để kiểm tra bổ sung. Groq đạt **F1 = 0,951** trên bộ khó, nhưng p95 khoảng **2,47 giây**, nên bước kiểm tra này cần chạy bất đồng bộ nếu hệ thống phải phản hồi ban đầu trong 50 ms. Em cũng sẽ kiểm tra một mẫu tin ngoài khoảng trên để phát hiện trường hợp BoW đoán sai dù rất tự tin.

## 7. AI log và điều đã học

| Phần | Em hỏi gì? | AI hỗ trợ gì? | Em đã sửa hoặc kiểm chứng gì? |
|---|---|---|---|
| **4 – Cấu hình API** | Phân biệt cấu hình tên Secret và giá trị API key thực tế trong môi trường Colab. | Tên Secret (ví dụ: `labngay01`) đóng vai trò làm biến định danh lưu trong môi trường Colab, trong khi ô Value sẽ lưu giữ mã API key thực tế từ dịch vụ Groq. | Cấu hình tham số `LLM_PROVIDER = "groq"` và thực thi hàm `list_models()`. Hệ thống truy xuất thành công danh sách 11 model khả dụng, bao gồm cả hai phiên bản 20B và 120B. |
| **4.1 – Gọi API** | Lý do tích hợp hệ thống Groq thông qua giao diện khởi tạo của thư viện `OpenAI()`. | Thư viện `OpenAI()` được cấu hình lại địa chỉ base URL để định hướng các yêu cầu HTTP/REST trực tiếp tới API của Groq, đồng thời thiết lập hàm `call_openai` xử lý truyền nhận dữ liệu. | Tiến hành sửa lỗi logic ở điều kiện gọi nhầm sang Anthropic, sau đó kích hoạt kiểm thử với mô hình Groq 20B. Kết quả mô hình phản hồi chính xác bằng tiếng Việt với độ trễ ghi nhận là **0,34 giây**. |
| **4.2–4.4 – Đọc nhãn** | Phương pháp xử lý chuỗi đầu ra để tránh nhận diện sai từ khóa `"not spam"` thành `"spam"`. | Xây dựng hàm `parse_label` thực hiện so khớp chính xác toàn bộ chuỗi nhãn đầu ra (exact match), đồng thời tích hợp biến đếm `n_invalid` đối với các phản hồi không đúng định dạng. | Chạy thử nghiệm phân loại trên tập dữ liệu 40 tin nhắn. Kết quả ghi nhận `invalid=0`, xác nhận toàn bộ 40 phản hồi đều được hệ thống trích xuất nhãn chuẩn xác. |
| **4.A–4.B – So sánh** | Đánh giá ảnh hưởng của kích thước mô hình và kỹ thuật Few-shot prompting đối với chỉ số F1-score. | Áp dụng phương pháp thực nghiệm kiểm soát biến: giữ nguyên tập dữ liệu 40 tin nhắn kiểm thử và tiến hành thay đổi độc lập giữa quy mô mô hình (20B vs 120B) hoặc cách thiết lập prompt. | Đã đo lường hiệu năng của mô hình 20B và 120B ở cấu hình Zero-shot, tiếp đó bổ sung 4 ví dụ minh họa (Few-shot) cho 20B. Kết quả cho thấy cả 3 phương án cấu hình đều đạt chỉ số **F1 = 0,930**. |

**Điều em học được:** Trước đây em nghĩ F1 cao trên tập test là đủ để chọn mô hình. Qua bài này, em thấy một mô hình có thể đạt F1 = 1,000 trên 40 tin quen thuộc nhưng chỉ đạt 0,368 trên 50 tin khó. Khi chọn bộ lọc spam, em sẽ xem cả độ trễ, số tin thường bị chặn nhầm và những kiểu lừa đảo mô hình bỏ sót.
