
# PHƯƠNG PHÁP NGHIÊN CỨU

## 1. Đề tài
Phát hiện tin giả sử dụng Word2Vec và BERT,
kết hợp các mô hình phân loại Naive Bayes,
CNN và RNN.

## 2. Dữ liệu
Sử dụng bộ dữ liệu WELFake Dataset.

Dữ liệu gồm các thuộc tính:
- title: Tiêu đề bài báo.
- text: Nội dung bài báo.
- subject: Chủ đề.
- date: Ngày đăng.
- label: Nhãn phân loại.

Quy ước nhãn:
- 1: Fake News.
- 0: Real News.

## 3. Tiền xử lý dữ liệu
- Kiểm tra dữ liệu thiếu.
- Xử lý dữ liệu trùng lặp.
- Làm sạch văn bản.
- Chuẩn hóa văn bản.
- Chia dữ liệu Train/Test.

## 4. Phương pháp biểu diễn văn bản
- Word2Vec: Biểu diễn từ dưới dạng vector.
- BERT: Biểu diễn ngữ cảnh của từ trong câu.

## 5. Mô hình phân loại
- Naive Bayes.
- CNN.
- RNN.

## 6. Đánh giá mô hình
Sử dụng các chỉ số:
- Accuracy.
- Precision.
- Recall.
- F1-score.
- Confusion Matrix.

## 7. So sánh kết quả
So sánh hiệu quả của các phương pháp
biểu diễn văn bản và mô hình phân loại
để lựa chọn phương pháp phù hợp.
