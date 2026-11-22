# Checkpoint phân đoạn đang dùng

Ứng dụng CBIR hiện dùng DeepLabV3+ 384 v2 đã chọn trong `D:\CSDLDPT\cau_3\active_deeplab_model.json`:

`D:\CSDLDPT\cau_3\deeplabv3plus_384_v2_finetuned\best_deeplabv3plus.pt`

Checkpoint này dự đoán mask cho **cả 1.000 ảnh kho và ảnh truy vấn mới**. Worker tại `D:\CSDLDPT\cau_3\cbir_deeplabv3plus_worker.py` gọi `mask_postprocessing.py` với ngưỡng 0,45 và các tham số trong `flower_cbir/config/pipeline.json`. Ứng dụng giữ thành phần liên thông lớn nhất trước khi trích rút 163 đặc trưng. Mask đã duyệt ở `D:\CSDLDPT\final_segmentation_v2` chỉ dùng làm nhãn tham chiếu để đánh giá chất lượng phân đoạn.

Mã ứng dụng kiểm tra SHA256 của checkpoint, cấu hình model, worker và hậu xử lý khi nạp phiên bản chỉ mục. Trọng số không được commit vào Git.

Checkpoint `deeplabv3_flower_v3.pt` trước đây thuộc thí nghiệm DeepLabV3 ResNet50 và **không phải** model đang dùng.

Khi chọn checkpoint khác, cần kiểm tra lại luồng query và xuất bản một phiên bản chỉ mục có contract tương ứng.
