# PHÁT HIỆN TIN GIẢ SỬ DỤNG WORD2VEC VÀ BERT

Dự án nghiên cứu phát hiện tin giả bằng biểu diễn văn bản Word2Vec và BERT, kết hợp các mô hình phân loại Naive Bayes, CNN và RNN.

## Mục tiêu
- Khám phá và tiền xử lý dữ liệu tin tức.
- Xây dựng đặc trưng văn bản bằng Word2Vec và BERT.
- Huấn luyện, so sánh các mô hình phân loại.
- Đánh giá bằng Accuracy, Precision, Recall, F1-score và Confusion Matrix.

## Cấu trúc repository
- `data/`: hướng dẫn và vị trí lưu dữ liệu.
- `notebooks_01_data_exploration.ipynb`: khám phá dữ liệu.
- `notebooks_02_preprocessing.ipynb`: tiền xử lý dữ liệu.
- `notebooks_03_word2vec_training.ipynb`: Word2Vec.
- `notebooks_04_naive_bayes.ipynb`: Naive Bayes.
- `src/`: mã nguồn Python dùng chung.
- `models/`: hướng dẫn lưu mô hình.
- `results/`: kết quả thực nghiệm.
- `docs/`: tài liệu dự án.
- `requirements.txt`: thư viện cần cài đặt.

## Dữ liệu
Dự án dự kiến sử dụng ISOT Fake News Dataset và WELFake Dataset. Xem [data/README.md](data/README.md) để biết nguồn tải và vị trí lưu.

**Lưu ý:** Không đưa CSV dung lượng lớn lên GitHub. Hãy lưu dữ liệu trên Google Drive/Kaggle và nạp vào Google Colab theo đường dẫn được cấu hình trong notebook.

## Cài đặt
```bash
pip install -r requirements.txt
```

## Thực hiện
Mở notebook bằng Google Colab hoặc Jupyter và chạy theo thứ tự từ 01 đến 04. Kiểm tra đường dẫn dữ liệu trước khi chạy.

## Đánh giá
Chỉ đưa số liệu thực nghiệm đã chạy và kiểm chứng vào báo cáo.

## Tác giả
Sinh viên thực hiện dự án – Trường Đại học Quy Nhơn.
