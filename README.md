# email-nlp

## Enron Spam — Kaggle Results Explorer

Dashboard tĩnh để trình bày và kiểm tra kết quả của notebook Kaggle phân loại email Enron. Trang còn có khu vực demo cho phép dán nội dung email hoặc tải ảnh chụp email để nhận dạng chữ (OCR) và phân loại spam/ham ngay trên trình duyệt.

## Chạy local

```bash
python3 -m http.server 4173 -d dist
```

Sau đó mở `http://localhost:4173`.

## Cấu trúc

```text
dist/                 Website sẵn sàng để deploy
  index.html          Khung giao diện
  styles.css          Design system và responsive layout
  app.js              Tương tác, biểu đồ và tra cứu dự đoán
  data/results.json   Dữ liệu đã tối ưu cho frontend
  assets/             Biểu đồ gốc từ Kaggle
scripts/
  build-data.mjs      Chuyển output CSV/JSON thành dữ liệu frontend
source-data/          Output nguồn chọn lọc từ Kaggle
```

## Nguồn dữ liệu

- `metrics.csv`: chỉ số của 3 mô hình.
- `lstm_history.csv`: lịch sử train/validation.
- `state.json`: thống kê tiền xử lý, chia tập và vocabulary.
- `test_predictions.csv`: 5.058 dự đoán trên tập test.

Các chỉ số, biểu đồ và xác suất trong khu vực tra cứu tập test được lấy trực tiếp từ output Kaggle. Khu vực **Thử phân loại** là một demo riêng: nó kết hợp thống kê Multinomial Naive Bayes nhẹ từ subject có nhãn trong `test_predictions.csv` với nhóm tín hiệu spam/ham phổ biến bằng tiếng Anh và tiếng Việt; ảnh được OCR bằng Tesseract.js. Demo này không phải mô hình LSTM gốc và không dành cho môi trường production.

OCR tải Tesseract.js cùng gói ngôn ngữ Anh + Việt từ CDN ở lần sử dụng đầu tiên, nên tính năng tải ảnh cần kết nối Internet. Nội dung và ảnh không được gửi tới backend của ứng dụng.
