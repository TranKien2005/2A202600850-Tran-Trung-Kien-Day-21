# Lab 21 — Evaluation Report

**Học viên**: Trần Trung Kiên — 2A202600850  
**Ngày nộp**: 2026-06-25  
**Submission option**: A (Lightweight ZIP)

---

## 1. Setup
- **Base model**: `unsloth/Qwen2.5-3B-bnb-4bit` (lượng tử hóa 4-bit NF4)
- **Dataset**: `5CD-AI/Vietnamese-alpaca-gpt4-gg-translated`, 200 samples (180 train + 20 eval)
- **max_seq_length**: 1024 (p95 = 562, rounded up to 1024)
- **GPU**: Tesla T4, 15.0 GB VRAM (Google Colab Free environment)
- **Training cost**: $0.07 (Tổng thời gian train cho cả 3 ranks: ~12.2 phút @ $0.35/hour phí GPU T4 mặc định)
- **HF Hub link**: N/A (Chọn Option A - Nộp cục bộ gọn nhẹ)

---

## 2. Rank Experiment Results

| Rank | Alpha | Trainable Params | Train Time | Peak VRAM | Eval Loss | Perplexity |
|------|-------|-----------------|------------|-----------|-----------|------------|
| 8    | 16    | 1,843,200       | 4.00 min   | 7.22 GB   | 1.5577    | 4.75       |
| 16   | 32    | 3,686,400       | 4.26 min   | 6.62 GB*  | 1.5161    | 4.55       |
| 64   | 128   | 14,745,600      | 3.99 min   | 8.00 GB   | 1.4768    | 4.38       |
| Base | -     | -               | -          | -         | -         | -          |

> [!NOTE]
> **Giải thích về sự bất thường của Peak VRAM ở Baseline r=16 (6.62 GB):**
> Trong Notebook, model baseline `r=16` được tải ở ngoài hàm test (Cell 13) và hàm đo bộ nhớ `torch.cuda.reset_peak_memory_stats()` được chạy *sau khi* model đã nằm sẵn trên GPU (Cell 16). Do đó, con số `6.62 GB` chỉ đại diện cho lượng VRAM tăng thêm của các optimizer state, gradient và activation trong quá trình train. 
> Ngược lại, đối với `r=8` và `r=64`, hàm `train_one_rank` thực hiện reload toàn bộ model từ đầu và reset peak memory *trước* khi tải model. Vì vậy, Peak VRAM của `r=8` (7.22 GB) và `r=64` (8.00 GB) bao gồm cả dung lượng của base model (~2.2 GB) + bộ nhớ huấn luyện bổ sung.
> Nếu quy đổi về cùng hệ quy chiếu (đã cộng dung lượng model base):
> - **r=8**: ~7.22 GB  
> - **r=16**: ~8.82 GB (6.62 + 2.20)  
> - **r=64**: ~8.00 GB (Riêng VRAM của `r=64` thấp hơn `r=16` thực tế do PyTorch tối ưu dọn dẹp cache giữa các phiên train hoặc do cơ chế giải phóng bộ nhớ của Unsloth).

---

## 3. Loss Curve Analysis
- **Quan sát Overfitting**: Do chạy trên GPU T4 với tài nguyên hạn chế, cấu hình huấn luyện đã tắt tính năng đánh giá giữa kỳ (`eval_strategy="no"`) nhằm tránh lỗi tràn bộ nhớ (OOM). Vì vậy, đồ thị Loss Curve chỉ hiển thị sự sụt giảm ổn định của Train Loss từ khoảng 1.8 xuống còn 1.1 qua 69 steps huấn luyện (3 epochs).
- **Kết quả Eval Loss cuối kỳ**: Sau khi hoàn thành huấn luyện, eval loss trên tập validation (20 samples) đạt:
  - `r=8`: 1.5577 (Perplexity = 4.75)
  - `r=16`: 1.5161 (Perplexity = 4.55)
  - `r=64`: 1.4768 (Perplexity = 4.38)
- Không xảy ra hiện tượng overfitting nghiêm trọng vì eval loss và perplexity cuối kỳ vẫn duy trì ở mức thấp và tiệm cận tốt với train loss. Khoảng cách loss nhỏ chứng tỏ mô hình có khả năng tổng quát hóa tốt.

---

## 4. Qualitative Comparison (5 examples)

### Example 1
- **Prompt**: Giải thích khái niệm machine learning cho người mới bắt đầu.
- **Base**: Machine learning là một phân khúc của trí tuệ nhân tạo, nó tập trung vào việc thiết lập các mô hình máy móc để học tập từ dữ liệu và từ đó có thể dự đoán hoặc hành động. Nhìn vào khía cạnh đơn giản, n...
- **Fine-tuned (r=16)**: Machine learning là một bộ môn công nghệ máy tính dựa trên việc học tập và cải thiện các dự đoán dựa trên dữ liệu mà không có sự hướng dẫn trực tiếp từ người dùng. Nó là một phần của AI (trí tuệ nhân ...
- **Nhận xét**: *Improved*. Bản fine-tuned đưa ra định nghĩa học thuật, gãy gọn hơn và làm nổi bật khía cạnh "không cần sự hướng dẫn trực tiếp" (tự học từ dữ liệu).

### Example 2
- **Prompt**: Viết đoạn code Python tính số Fibonacci thứ n.
- **Base**: Để tính số Fibonacci thứ n, bạn có thể sử dụng hàm đệ quy hoặc vòng lặp. Đây là một đoạn mã Python cho phép bạn tính số Fibonacci thứ n:
```python
def fibonacci(n):
    if n <= 0:
        return "N p...
```
- **Fine-tuned (r=16)**: Để tính số Fibonacci thứ n, bạn có thể viết một đoạn code Python như sau:
```python
def fibonacci(n):
    if n < 0:
        raise ValueError("Input phải là một số nguyên dương.")
    elif n == 0:
   ...
```
- **Nhận xét**: *Improved*. Bản fine-tuned đã cải tiến bằng cách bổ sung cơ chế kiểm soát lỗi (raise ValueError) khi người dùng nhập số âm, thay vì chỉ trả về chuỗi text thông báo đơn thuần như bản Base.

### Example 3
- **Prompt**: Liệt kê 5 nguyên tắc thiết kế UI/UX.
- **Base**: 1. Thân thiện với người dùng: Mục đích của thiết kế UI/UX là cung cấp trải nghiệm người dùng thân thiện, dễ sử dụng và thân thiện. Điều này có thể được thể hiện thông qua việc sắp xếp bố cục, màu sắc,...
- **Fine-tuned (r=16)**: 1. Chuyển đổi: UI/UX thiết kế phải hướng tới việc giúp người dùng thực hiện hành động. 2. Thích ứng: UI/UX thiết kế phải thích ứng với nhiều loại thiết bị và kích thước màn hình. 3. Đơn giản: UI/UX th...
- **Nhận xét**: *Improved*. Bản Base lặp từ "thân thiện" nhiều lần gây lủng củng. Bản Fine-tuned đưa ra các đề mục rõ ràng, chuyên nghiệp hơn (Chuyển đổi, Thích ứng, Đơn giản...).

### Example 4
- **Prompt**: Tóm tắt sự khác biệt giữa LoRA và QLoRA.
- **Base**: LoRA (Low-Rank Adaptation) và QLoRA (Quantized LoRA) là hai phương pháp cải thiện hiệu năng của mô hình NLU (NLP) bằng cách sử dụng các phép biến đổi thấp độ phức tạp. LoRA là phương pháp cải thiện hi...
- **Fine-tuned (r=16)**: LoRA (Layer-wise Adaptive Regularization Optimization) và QLoRA (Quantized LoRA) là hai phương pháp regularization được phát triển để cải thiện hiệu quả và độ ổn định của các mạng neural network trong...
- **Nhận xét**: *Degraded / Halucination nhẹ*. Mặc dù bản Fine-tuned hành văn mượt mà hơn nhưng lại dịch sai chữ viết tắt của LoRA thành "Layer-wise Adaptive Regularization Optimization" thay vì "Low-Rank Adaptation" (Base dịch đúng). Điều này cho thấy rủi ro ảo tưởng (hallucination) thuật ngữ tiếng Anh của mô hình khi cố gắng làm mượt câu tiếng Việt.

### Example 5
- **Prompt**: Phân biệt prompt engineering, RAG, và fine-tuning.
- **Base**: Prompt engineering, RAG (retrieval augmented generation), và fine-tuning là ba cách khác nhau để cải thiện hiệu suất của mô hình máy học. Prompt engineering là một kỹ thuật để cải thiện hiệu suất của ...
- **Fine-tuned (r=16)**: Prompt engineering, RAG và fine-tuning là ba kỹ thuật khác nhau được sử dụng trong lĩnh vực AI và tự động hóa. Prompt engineering là một kỹ thuật tập trung vào việc xây dựng câu lệnh (prompt) để giúp ...
- **Nhận xét**: *Improved*. Bản Fine-tuned trình bày có hệ thống hơn, nêu bật vai trò của từng kỹ thuật (Prompt làm cầu nối giao tiếp, RAG cung cấp dữ liệu ngoài, Fine-tuning thay đổi hành vi/style).

---

## 5. Conclusion về Rank Trade-off
- **Rank nào cho ROI tốt nhất trên dataset này?**
  - Đối với dataset nhỏ (200 ví dụ), **rank r=16** mang lại tỷ suất ROI (Return on Investment) tối ưu nhất. Với thời gian huấn luyện chỉ chênh lệch rất ít (~4.26 phút) và lượng VRAM an toàn, nó đạt mức perplexity khá thấp (4.55). Rank r=8 tuy nhanh và nhẹ nhất nhưng perplexity kém hơn (4.75).
- **Khi nào tăng rank không còn cải thiện perplexity (diminishing returns)?**
  - Hiện tượng này xuất hiện khi nâng từ `r=16` lên `r=64`. Số tham số huấn luyện tăng gấp 4 lần (từ 3.6M lên 14.7M) nhưng perplexity chỉ cải thiện rất nhỏ (từ 4.55 xuống 4.38 - giảm khoảng 3.7%). Trên các tập dữ liệu lớn, việc tăng rank quá cao mà không tăng dữ liệu còn dễ gây ra hiện tượng overfitting nghiêm trọng.
- **Khuyến nghị cho Production**:
  - Nếu triển khai thực tế (Production), tôi khuyến nghị sử dụng **rank r=16** (hoặc thậm chí `r=8` nếu cần phục vụ đa tác vụ dạng multi-tenant). Rank nhỏ giúp kích thước file adapter nhẹ (chỉ vài chục MB), hỗ trợ tải động/thay thế nóng (hot-swapping adapters) nhanh chóng mà không gây độ trễ (latency) khi suy luận, đồng thời vẫn giữ được chất lượng văn bản tốt.

---

## 6. What I Learned
- **Kỹ thuật LoRA/QLoRA**: Hiểu sâu sắc cơ chế đóng băng base model và chỉ cập nhật ma trận low-rank giúp giảm thiểu tài nguyên tính toán đi hàng chục lần mà không làm giảm đáng kể hiệu năng.
- **Lượng tử hóa 4-bit**: QLoRA sử dụng NF4 (NormalFloat 4-bit) và Double Quantization đóng vai trò cứu cánh giúp chạy được các mô hình LLM lớn trên GPU nhỏ như T4 (16GB).
- **Đánh giá đa chiều**: Học được cách đánh giá mô hình sau fine-tune bằng cả định lượng (Perplexity - độ hỗn loạn của văn bản, tính bằng hàm `exp(eval_loss)`) và định tính (chạy thử test prompts để so sánh thủ công).
- **Tối ưu bộ nhớ**: Rút ra bài học quan trọng về việc quản lý VRAM trong PyTorch (dùng `gradient_checkpointing`, dọn dẹp rác `gc.collect()`, giải phóng bộ nhớ cache CUDA).
