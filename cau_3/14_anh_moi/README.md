# Bộ 14 Ảnh Hoa Mới — Kiểm Thử Mô Hình DeepLabV3+ (Câu 3)

Thư mục này lưu trữ riêng biệt 14 ảnh hoa mới do người dùng cung cấp dùng để thử nghiệm mô hình DeepLabV3+ (`deeplabv3plus_384_v2_finetuned`) và quy trình hậu xử lý chuẩn.

## Cấu trúc thư mục
- `images/`: 14 ảnh hoa gốc (định dạng PNG 384 × 384 RGB).
- `masks/`: 14 mặt nạ tách hoa nhị phân (PNG 384 × 384 uint8 {0, 255}) do DeepLabV3+ sinh ra.
- `cutouts/`: 14 ảnh hoa tách nền trong suốt (PNG RGBA).
- `overlays/`: 14 ảnh phủ đường bao và mặt nạ lên ảnh gốc.

## Danh mục 14 bông hoa
| STT | Mã tệp | Tên loài hoa | Màu sắc chủ đạo | Kích thước |
|:---:|:---|:---|:---|:---:|
| 01 | `flower_01_huong_duong.png` | Hoa Hướng Dương (Sunflower 1) | Vàng | 384 × 384 |
| 02 | `flower_02_cuc_trang.png` | Hoa Cúc Trắng (White Daisy) | Trắng | 384 × 384 |
| 03 | `flower_03_cuc_bach_nhat.png` | Hoa Cúc Bách Nhật (Pink Zinnia) | Hồng | 384 × 384 |
| 04 | `flower_04_van_tho.png` | Hoa Vạn Thọ (Orange Marigold) | Cam | 384 × 384 |
| 05 | `flower_05_canh_buom_tim.png` | Hoa Cánh Bướm Tím (Purple Cosmos) | Tím | 384 × 384 |
| 06 | `flower_06_dam_but_do.png` | Hoa Dâm Bụt Đỏ (Red Hibiscus 1) | Đỏ | 384 × 384 |
| 07 | `flower_07_huong_duong_2.png` | Hoa Hướng Dương (Sunflower 2) | Vàng | 384 × 384 |
| 08 | `flower_08_hong_do_nhung.png` | Hoa Hồng Đỏ Nhung (Red Rose) | Đỏ nhung | 384 × 384 |
| 09 | `flower_09_phong_lan_tim.png` | Hoa Phong Lan Tím (Pink Orchid) | Tím hồng | 384 × 384 |
| 10 | `flower_10_cuc_dai_doa.png` | Hoa Cúc Đại Đóa (Yellow Chrysanthemum) | Vàng | 384 × 384 |
| 11 | `flower_11_dam_but_do_xoe.png` | Hoa Dâm Bụt Đỏ Xòe (Red Hibiscus 2) | Đỏ tươi | 384 × 384 |
| 12 | `flower_12_sung_tim.png` | Hoa Súng Tím (Purple Water Lily) | Tím / vàng | 384 × 384 |
| 13 | `flower_13_su_trang.png` | Hoa Sứ Trắng Tâm Vàng (White Plumeria) | Trắng / vàng | 384 × 384 |
| 14 | `flower_14_canh_buom_hong.png` | Hoa Cánh Bướm Hồng (Pink Cosmos) | Hồng phấn | 384 × 384 |