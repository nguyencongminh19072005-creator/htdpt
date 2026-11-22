# DeepLabV3+ 384 × 384 — kết quả fine tune ngày 25/10/2026

## Dữ liệu và cách đánh giá

- Đầu vào/đầu ra model mới: **384 × 384**, không resize ảnh từ 384 xuống 256 khi dự đoán.
- Ground truth: 112 mask đã sửa trong `final_segmentation_v2`; 91 ảnh train, 7 ảnh validation, 14 ảnh kiểm tra giữ riêng. Còn 17 ảnh mới chưa có ground truth, chỉ xem trực quan.
- Checkpoint gốc 256 được dùng để khởi tạo trọng số; các checkpoint mới dự đoán trực tiếp ở 384 × 384.
- Kiến trúc trong mã nguồn là bản DeepLabV3+ gọn, encoder tự huấn luyện, không có backbone pretrained. Máy hiện tại chạy CPU; không có CUDA hoặc trọng số encoder pretrained trong cache.
- Chỉ mask đã sửa được đưa vào fine tune. Không sử dụng 14 ảnh kiểm tra hoặc 17 ảnh mới để cập nhật trọng số. Tuy nhiên 14 ảnh đã được xem để so sánh nhiều thí nghiệm, nên số liệu này không còn là phép đo cuối hoàn toàn độc lập.

## Các lần thử

| Lần thử | Validation IoU (7 ảnh) | IoU trên 14 ảnh giữ riêng | Nhận xét |
| --- | ---: | ---: | --- |
| 384 ban đầu | 0,955222 | 0,946631 | Mốc 384 đầu tiên |
| Tiếp tục train | 0,956083 | 0,947076 | Cải thiện ít |
| Model 256 hướng dẫn | 0,956624 | 0,948282 | Validation cao nhất trong các lần train |
| Ưu tiên pixel hoa bị bỏ sót | 0,956587 | 0,949578 | Tăng độ nhạy |
| Hình học + ưu tiên pixel hoa (`augmented`) | 0,955809 | **0,951063** | Checkpoint 384 đang dùng |
| Biến đổi màu mạnh | 0,953710 | 0,945140 | Giảm chất lượng |
| Model 256 hướng dẫn mạnh hơn | 0,955461 | 0,950081 | Không vượt bản đang dùng |
| Tăng trọng số vùng biên 7 pixel | 0,955796 | 0,951078 | Chênh lệch 0,000014, không có ý nghĩa thực tế |

Checkpoint 256 trước đây đạt **0,967624** trên cùng 14 ảnh. Vì vậy checkpoint 384 hiện tại chưa tốt hơn model cũ ở tập này. Các số liệu từ chỉ 7 ảnh validation và 14 ảnh giữ riêng có độ bất định cao; không nên xem chênh lệch vài phần mười nghìn là cải thiện chắc chắn.

Thử dự đoán qua bốn hướng quay và chọn ngưỡng bằng validation nâng validation IoU lên 0,958500, nhưng IoU trên 14 ảnh giảm còn 0,949384. Kết hợp hai checkpoint cũng chỉ đạt 0,951284 trên 14 ảnh khi chọn cấu hình từ validation, đổi lại chi phí suy luận cao hơn. Vì vậy hai cách này không được đưa vào luồng dự đoán đang dùng.

## Checkpoint được giữ

`deeplabv3plus_384_augmented_20261025/best_deeplabv3plus.pt`

`active_deeplab_model.json` vẫn trỏ đến checkpoint này. Không thay đổi dữ liệu đặc trưng CBIR hoặc PostgreSQL trong lần fine tune này.

## Lỗi còn lại và dữ liệu cần bổ sung

Ba ảnh khó nhất trong tập giữ riêng:

| Ảnh | IoU 384 | Quan sát |
| --- | ---: | --- |
| `image_06732.png` | 0,873744 | Hoa lan: còn nhầm vùng nền sát cánh và thiếu một số mép mảnh |
| `image_02071.png` | 0,894774 | Hoa trắng có vùng tối: thiếu nhiều phần cánh và vùng giữa |
| `image_04606.png` | 0,901425 | Cánh tím xanh: thiếu các vùng tối và mép cánh |

Muốn tăng chất lượng đáng kể cần thêm mask **được kiểm tra thủ công** từ những ảnh hiện chưa nằm trong 112 mask đã sửa, nhất là hoa trắng trong bóng, hoa có cánh phức tạp và các khe mảnh giữa cánh. Giữ 14 ảnh hiện tại ngoài train; đồng thời cần một tập test mới có ground truth chưa từng dùng để quyết định thí nghiệm trước khi kết luận model mới tốt hơn. Ảnh minh họa lỗi: `deeplabv3plus_384_augmented_20261025/heldout_diagnostics.jpg`.

## Cập nhật về mask đã xem và giữ nguyên

Người dùng xác nhận các mask v2 còn lại đã được xem và chấp nhận nguyên trạng. Bản kê `reviewed_unchanged_384.json` ghi 871 mask trong bộ `image_384` hiện tại lấy từ `final_segmentation_v2_archive_before_curated_20261025`: 687 ảnh train, 93 validation, 91 thuộc nhóm kiểm tra cũ. Có thêm 17 ảnh gốc trong bản lưu nhưng không còn nằm trong `image_384` hiện tại nên không đưa vào bản kê.

**Các số liệu ở trên là kết quả của các lần train trước, chỉ dùng 91 mask train đã sửa.** Tại thời điểm ghi nhận ban đầu, có thêm 687 mask train đã xem và giữ nguyên. Lần train tiếp theo ở mục dưới đã dùng tùy chọn `--include-reviewed-unchanged`; validation vẫn dùng 7 mask đã sửa và phép kiểm tra vẫn dùng 14 mask đã sửa. Không dùng mask v3 mới chưa được xem làm nhãn. Trang sửa mask ở `http://127.0.0.1:8001/` mở mask v2 đã duyệt để bạn chỉnh tiếp; khi lưu một ảnh, mask sửa được thêm vào `final_segmentation_v2`.

## Lần train với mask đã duyệt nguyên trạng

Sau khi người dùng sửa thêm bốn mask train, bản nhãn được chốt vào `deeplabv3plus_384_reviewed_20261025/reviewed_masks_snapshot`: 95 mask train đã sửa tay và 683 mask train đã xem, giữ nguyên từ bản lưu v2. Mỗi epoch dùng 95 mask sửa tay ba lần và 683 mask giữ nguyên một lần, tổng 968 lượt ảnh. Validation vẫn là 7 mask đã sửa; kiểm tra là 14 mask đã sửa. Những ảnh sửa sau lúc chốt nhãn không tự đi vào lần train này.

Checkpoint khởi tạo là bản 384 `augmented`, đầu vào/đầu ra đều 384 × 384. Fine tune dùng geometric augmentation, AdamW với learning rate `3e-5`, Tversky ưu tiên bỏ sót hoa `0,6`; dừng sớm ở epoch 9, chọn epoch 5 theo validation.

| Model 384 | Validation IoU (7 ảnh) | IoU trên 14 ảnh có nhãn |
| --- | ---: | ---: |
| Trước lần train này | 0,955809 | 0,951063 |
| Sau lần train này | **0,956438** | **0,956144** |

Model 256 trước đây đạt 0,967624 trên cùng 14 ảnh. Tập 14 ảnh đã được xem trong nhiều thí nghiệm, nên cần tập test mới có nhãn và chưa từng dùng để lựa chọn cách train trước khi khẳng định mức cải thiện tổng quát.

Checkpoint mới đang hoạt động qua `active_deeplab_model.json`. Kết quả dự đoán 1.000 ảnh của nó nằm riêng trong `final_segmentation_v4_384`; đây là mask **dự đoán**, không ghi đè các mask sửa tay trong `final_segmentation_v2`. IoU 0,976100 trong manifest v4 chỉ đo mức trùng với mask v1, không phải độ chính xác so với ground truth.
