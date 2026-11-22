# Công cụ trích rút, CSV và PostgreSQL

`full_163.py` giữ các công cụ `extract`, `check`, `load`, `all`, nhưng **không còn một bản thuật toán riêng**. Nó import cùng `flower_cbir.core.features.extract_all_features` và dùng chung pipeline DeepLabv3 với chương trình truy vấn.

## Các lệnh

Từ thư mục dự án:

```powershell
# Chỉ trích rút và export, chưa ghi DB
.venv/Scripts/python.exe trich_rut_minh/full_163.py extract

# Kiểm tra CSV, hash nguồn, chạy lại model và đối chiếu từng vector
.venv/Scripts/python.exe trich_rut_minh/full_163.py check

# Import: kiểm tra tương thích trước, rồi đồng bộ toàn bộ các bảng trong transaction
.venv/Scripts/python.exe trich_rut_minh/full_163.py load

# Trích rút, export và import
.venv/Scripts/python.exe trich_rut_minh/full_163.py all
```

Có thể chạy script bằng đường dẫn tuyệt đối từ working directory khác. Cần giữ package `flower_cbir` cùng thư mục gốc dự án; không copy riêng script và tự chỉnh lại công thức.

Các tùy chọn `--image-dir`, `--mask-dir`, `--csv`, `--manifest`, `--expected-count`, `--host`, `--port`, `--database`, `--user` được giữ. Ví dụ:

```powershell
.venv/Scripts/python.exe trich_rut_minh/full_163.py all `
  --image-dir '../postgres_data/image_384' `
  --csv trich_rut_minh/features_163.csv `
  --manifest trich_rut_minh/extraction_manifest_163.json
```

Mặc định không cần `--mask-dir`: các mask DeepLabv3 được tạo/cached bởi cùng model của query. Nếu truyền một thư mục mask, từng mask sau làm sạch phải khớp chính xác với dự đoán của checkpoint đang cấu hình; mask cũ của thuật toán/model khác bị từ chối. Để xử lý lại dữ liệu đã có trong hệ thống này, truyền thư mục ảnh và để pipeline tạo mask mới, không truyền mask cũ từ SAM/GrabCut hoặc model khác.

## Định dạng và tính toàn vẹn

- CSV giữ metadata `filename,image_sha256,mask_sha256,flower_pixels`, tiếp theo là 163 cột đã đặt tên. Thứ tự F1 | F2 | F3 | F4 không đổi.
- Manifest là phần bắt buộc của batch: contract thuật toán, hash checkpoint, runtime, hash CSV, từng nguồn ảnh/mask, phiên bản và normalizer.
- Đầu ra float32 ghi bằng `.17g`, đọc lại không mất bit. Không import CSV cũ kiểu phân vị F2/LBP nội bộ, dù header cũng có 163 chiều.
- `load` xác minh nguồn, dự đoán lại mask, trích lại vector và so bằng `np.array_equal` trước khi ghi DB. Sau ghi, đọc lại PostgreSQL để so cả vector thô và vector chuẩn hóa.
- Import cập nhật cả `flower_features`, `feature_normalization`, `normalized_features` cùng phiên bản hoạt động; không chỉ nạp các cột feature thô như script cũ.
- Import phải gồm toàn bộ các ảnh đang có trong gallery; batch thiếu ảnh bị từ chối. Không tự xóa ảnh/hàng cũ để làm đủ số lượng.

Thông tin PostgreSQL dùng chung `database.local.json`/biến môi trường PG* của dự án. Không còn phụ thuộc `python-dotenv`, `.env` riêng hoặc một PDF hardcode không tồn tại.

API `read_pair(image_path,mask_path)` và `extract_one(rgb,mask)` vẫn có để dùng với dữ liệu đã chuẩn hóa; `extract_one` chỉ chia vector trả về từ hàm chung thành bốn nhóm, không triển khai lại F1–F4. API với mask chỉ phục vụ đối chiếu cùng đầu vào; một mask tự sửa không được xem là dự đoán DeepLabv3 hợp lệ khi import strict.
