# htdpt

Hệ Thống Cơ Sở Dữ Liệu Lưu Trữ và Tìm Kiếm Ảnh Bông Hoa (Content-Based Image Retrieval - CBIR)

Đề tài Bài tập lớn môn học **Hệ Cơ Sở Dữ Liệu Đa Phương Tiện (HCSDLDPT)** — Học viện Công nghệ Bưu chính Viễn thông (PTIT).

---

## 📌 Tổng quan Hệ thống

Hệ thống cung cấp giải pháp toàn diện cho bài toán quản lý, phân đoạn và truy hồi ảnh hoa theo nội dung:
1. **Phân đoạn ngữ nghĩa tự động (Câu 3)**: Sử dụng mô hình mạng nơ-ron tích chập sâu **DeepLabV3+** ($384 \times 384$) kết hợp bộ lọc hậu xử lý chuyên biệt (`mask_postprocessing`) để tách chính xác bông hoa khỏi nền phức tạp.
2. **Trích rút đặc trưng 163 chiều chuẩn mực**: Triển khai thuật toán trích rút 12 thuộc tính thuộc 4 nhóm đặc trưng:
   - **F1 (Màu sắc HSV toàn cục — 78 chiều)**: Biểu đồ kết hợp H–S ($72\text{D}$) và V ($6\text{D}$).
   - **F2 (Màu sắc Lab theo 3 vành bán kính — 18 chiều)**: Giá trị Mean và Std Lab trên 3 vùng phân vị.
   - **F3 (Kết cấu bề mặt LBP bất biến quay — 28 chiều)**: Biểu đồ đa tỷ lệ $\text{LBP}_{8,1}^{riu2}$ ($10\text{D}$) và $\text{LBP}_{16,2}^{riu2}$ ($18\text{D}$).
   - **F4 (Hình dạng & Cấu trúc hoa — 39 chiều)**: 4 chỉ số hình học vô hướng, phổ Fourier bán kính ($12\text{D}$), mô hình góc nội gradient IAP ($16\text{D}$), mô hình mật độ cạnh IEP ($7\text{D}$).
3. **Độ đo khoảng cách kết hợp**: So sánh biểu đồ xác suất bằng khoảng cách **Hellinger** ($D_1, D_3$) và so sánh giá trị số liên tục bằng khoảng cách **RMS biến đổi** ($D_2, D_4$), kết hợp trọng số người dùng tùy biến.
4. **Hệ thống CSDL & Web App hoàn chỉnh**: Lưu trữ trên PostgreSQL (7 bảng quản lý ảnh, đặc trưng, phiên bản index và lịch sử tìm kiếm) kết hợp giao diện Streamlit tương tác mượt mà.

---

## 📂 Cấu trúc Thư mục Dự án

```text
htdpt/
├── README.md                          # Giới thiệu và hướng dẫn tổng thể dự án
├── requirements.txt                   # Danh sách thư viện Python cần thiết
├── database.example.json              # Mẫu tệp cấu hình kết nối PostgreSQL
├── main.py                            # Điểm khởi chạy chính (--app, --index, --search)
│
├── flower_cbir/                       # [HỆ THỐNG CBIR HOÀN THÀNH]
│   ├── app.py                         # Giao diện tìm kiếm trực quan Streamlit
│   ├── settings.py                    # Cấu hình chung của ứng dụng
│   ├── config/
│   │   └── pipeline.json              # Cấu hình tham số toàn bộ pipeline
│   ├── core/                          # Lõi thuật toán CBIR
│   │   ├── distance.py                # Hàm đo khoảng cách Hellinger + RMS biến đổi
│   │   ├── segmentation.py            # Giao diện phân đoạn ảnh
│   │   ├── deeplab.py                 # Wrapper mô hình DeepLabV3+
│   │   ├── deeplab_worker.py          # Worker tiến trình chạy phân đoạn
│   │   ├── schema.py                  # Định nghĩa và kiểm thực vector 163D
│   │   ├── normalization.py           # Chuẩn hóa Z-score chọn lọc (F2 & F4)
│   │   └── features.py                # Tính toán 4 nhóm đặc trưng
│   ├── services/                      # Tầng dịch vụ hệ thống
│   │   ├── database.py                # Quản lý kết nối và truy vấn PostgreSQL
│   │   ├── search.py                  # Lõi xếp hạng Top-K tương đồng
│   │   ├── index_store.py             # Xuất bản và xác thực chỉ mục vào DB
│   │   └── verification.py            # Kiểm thử tính toàn vẹn hệ thống
│   └── evaluation/                    # Đánh giá độ đo truy hồi (Precision/Recall)
│
├── cau_3/                             # [CÂU 3: DEEPLABV3+ & HẬU XỬ LÝ CHUẨN]
│   ├── README.md                      # Hướng dẫn chi tiết Câu 3
│   ├── TRAINING_384_RESULTS.md        # Báo cáo kết quả huấn luyện mô hình
│   ├── mask_postprocessing.py         # Bộ lọc hậu xử lý chuẩn (threshold 0.45, repair_center, keep_nearby)
│   ├── active_deeplab_model.py        # Module nạp model kích hoạt
│   ├── active_deeplab_model.json      # Khai báo model kích hoạt mặc định
│   ├── predict_deeplab_1000.py        # Phân đoạn tự động cho 1.000 ảnh CSDL
│   ├── predict_new_flowers.py         # Phân đoạn cho ảnh hoa mới bất kỳ
│   ├── finetune_deeplabv3plus_v2.py   # Huấn luyện fine-tune DeepLabV3+
│   ├── train_flower_deeplabv3plus.py  # Định nghĩa kiến trúc mạng DeepLabV3+
│   ├── deeplabv3plus_384_v2_finetuned/# Checkpoint mô hình chính thức (mIoU 98.50%)
│   │   └── best_deeplabv3plus.pt      # Trọng số mô hình đã huấn luyện (0.93 MB)
│   └── 14_anh_moi/                    # Bộ 14 ảnh hoa mới dùng kiểm thử thực nghiệm
│       ├── images/                    # 14 ảnh hoa gốc 384x384
│       ├── masks/                     # 14 mặt nạ phân đoạn chuẩn
│       ├── cutouts/                   # 14 ảnh hoa tách nền trong suốt
│       ├── overlays/                  # 14 ảnh phủ viền đối chiếu
│       └── README.md                  # Danh mục thông tin chi tiết
│
├── trich_rut_minh/                    # [TRÍCH RÚT ĐẶC TRƯNG 163 CHIỀU]
│   ├── full_163.py                    # Thuật toán trích rút F1..F4 163 chiều (bản chuẩn CBIR)
│   ├── features_163.csv               # Bảng vector đặc trưng của 1.000 ảnh
│   ├── extraction_manifest_163.json   # Siêu dữ liệu kiểm tra toàn vẹn
│   └── README.md                      # Hướng dẫn sử dụng công cụ dòng lệnh
│
├── database/                          # [CƠ SỞ DỮ LIỆU POSTGRESQL]
│   └── schema.sql                     # Cấu trúc bảng CSDL chuẩn mực
│
└── tools/                             # Công cụ bổ trợ trích xuất & kiểm thử
    ├── export_deeplab_checkpoint.py
    ├── finetune_deeplab.py
    └── prepare_segmentation_data.py
```

---

## 🚀 Hướng dẫn Cài đặt & Khởi chạy

### 1. Cài đặt Môi trường
Khuyến nghị sử dụng Python 3.10 trở lên:
```bash
python -m venv .venv
source .venv/bin/activate       # Trên Linux/macOS
.venv\Scripts\activate          # Trên Windows

pip install -r requirements.txt
```

### 2. Cấu hình Cơ sở dữ liệu (PostgreSQL)
Tạo tệp cấu hình từ mẫu có sẵn:
```bash
cp database.example.json database.local.json
```
Mở `database.local.json` và điền thông tin kết nối PostgreSQL của bạn:
```json
{
  "host": "localhost",
  "port": 5432,
  "dbname": "flower_database",
  "user": "postgres",
  "password": "your_password",
  "schema": "public",
  "connect_timeout": 10
}
```

### 3. Khởi chạy Giao diện Tìm kiếm (Web App)
Khởi chạy ứng dụng Streamlit tìm kiếm ảnh hoa tương đồng:
```bash
python main.py --app
```
Giao diện sẽ tự động mở trên trình duyệt tại địa chỉ `http://localhost:8501`. Người dùng có thể:
- Tải lên ảnh hoa bất kỳ để tự động tách nền và trích rút vector.
- Điều chỉnh 4 thanh trượt trọng số $w_1, w_2, w_3, w_4$ theo nhu cầu tìm kiếm (Màu sắc, Bố cục vành, Kết cấu, Hình dạng).
- Nhận về Top 5 bông hoa tương đồng nhất kèm khoảng cách chi tiết và lịch sử truy vấn.

### 4. Chạy Phân đoạn Ảnh Mới với DeepLabV3+ (Câu 3)
Phân đoạn tự động cho một ảnh hoa mới với model chuẩn và hậu xử lý:
```bash
python cau_3/predict_new_flowers.py path/to/flower.jpg
```

### 5. Chạy Trích rút Đặc trưng 163 Chiều
Trích xuất đặc trưng cho tập dữ liệu:
```bash
python trich_rut_minh/full_163.py --help
```

---

## 🔬 Thông số Kỹ thuật & Độ chính xác

* **Kiến trúc phân đoạn**: DeepLabV3+ (Backbone ResNet thu nhỏ, `width=32`, `rates=(2, 4, 6)`), input/output $384 \times 384$.
* **Độ chính xác phân đoạn**:
  - **Validation mIoU**: $98.50\%$ (Dice: $99.24\%$)
  - **Held-out Test mIoU**: $98.18\%$ (Dice: $99.07\%$)
  - Kiểm thử thành công hoàn hảo trên $14/14$ ảnh hoa mới ngoài tập dữ liệu.
* **Độ đo tương đồng**:
  $$D = w_1 D_1 + w_2 D_2 + w_3 D_3 + w_4 D_4, \quad S = 1 - D \in [0, 1]$$
  - $D_1, D_3$: Khoảng cách Hellinger trên biểu đồ xác suất.
  - $D_2, D_4$: Khoảng cách RMS biến đổi trên vector số thực Z-score.
