# Phân đoạn hoa đang dùng cho CBIR

Pipeline đang hoạt động trong `D:\CSDLDPT_CBIR`:

1. Ảnh 384 × 384 đi qua DeepLabV3+ `width=32`, `rates=(2,4,6)`.
2. `mask_postprocessing.py` biến xác suất thành mask nhị phân với `threshold=0.45`, `opening=False`, `closing=False`, `fill_holes=True`, `max_hole_area=20`, `keep_nearby=True`, `repair_center=True`.
3. CBIR giữ thành phần liên thông lớn nhất, trích rút 163 đặc trưng, chuẩn hóa F2/F4 bằng Z-score rồi tìm kiếm trong PostgreSQL.

Checkpoint được chọn nằm ở `deeplabv3plus_384_v2_finetuned/best_deeplabv3plus.pt`. `active_deeplab_model.json` lưu đường dẫn và SHA-256. `active_deeplab_model.py` nạp kiến trúc `DeepLabV3Plus` từ `train_flower_deeplabv3plus.py`; `cbir_deeplabv3plus_worker.py` phục vụ cùng checkpoint và hậu xử lý cho cả ảnh kho lẫn ảnh truy vấn.

Mask sửa tay trong `D:\CSDLDPT\final_segmentation_v2` là nhãn tham chiếu để đánh giá. Mask hiện lưu trong PostgreSQL là dự đoán model sau hậu xử lý. Các checkpoint, mask v3/v5 và ảnh thử nghiệm cũ được giữ tại `D:\CSDLDPT\archive_legacy`.

Chạy ứng dụng từ `D:\CSDLDPT_CBIR`:

```powershell
& 'C:\Users\Admin\AppData\Local\Programs\Python\Python313\python.exe' main.py --app
```

Xem `D:\CSDLDPT_CBIR\README.md` để biết cách xây lại chỉ mục và kiểm tra đồng bộ.
