# Enron Spam — Kaggle Results Explorer

Dashboard tĩnh để trình bày và kiểm tra kết quả của notebook Kaggle phân loại email Enron.

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

Ứng dụng không chạy lại mô hình trong trình duyệt. Ô tìm kiếm chỉ tra cứu các mẫu có trong tập test, vì vậy mọi xác suất hiển thị đều lấy trực tiếp từ output Kaggle.
