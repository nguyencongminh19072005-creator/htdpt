"""
HỆ THỐNG WEB PHÂN ĐOẠN NGỮ NGHĨA ẢNH HOA DEEPLABV3+ (DEEPLAB SEGMENTATION WEB)
Chức năng: Ứng dụng Web độc lập phục vụ kiểm thử, trực quan hóa và nghiệm thu
           kết quả phân đoạn trung gian của mô hình DeepLabV3+ (Input -> Binary Mask 0/255 -> Segmented ROI -> Overlay).
Tác giả: Nhóm 9 - Nguyễn Công Minh (CSDL & CBIR)
"""
import io
import os
import sys
import time
from pathlib import Path

# Cấu hình đường dẫn thư mục gốc
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import cv2
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image
import streamlit as st

from flower_cbir.core.deeplab import run_deeplab
from flower_cbir.core.preprocess import load_image

# Thiết lập giao diện Streamlit rộng rãi và thẩm mỹ
st.set_page_config(
    page_title="DeepLabV3+ Flower Segmentation Web App",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS cho giao diện trực quan chuyên nghiệp
st.markdown("""
<style>
    .main-header {
        font-size: 2.1rem;
        font-weight: 700;
        color: #00E5FF;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        font-size: 1.05rem;
        color: #B0BEC5;
        margin-bottom: 1.5rem;
    }
    .metric-card {
        background-color: #1E293B;
        padding: 1rem;
        border-radius: 8px;
        border: 1px solid #334155;
        text-align: center;
    }
    .highlight-bit1 {
        color: #00E676;
        font-weight: bold;
    }
    .highlight-bit0 {
        color: #FF5252;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)

# Tiêu đề chính của ứng dụng
st.markdown('<div class="main-header">🌸 HỆ THỐNG TRỰC QUAN HÓA PHÂN ĐOẠN ẢNH HOA DEEPLABV3+</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="sub-header">Mô hình DeepLabV3+ Fine-tuned (mIoU 98.50%) | Xuất Mặt nạ Nhị phân '
    '<span class="highlight-bit1">Hoa trắng (255 / Bit 1)</span> - '
    '<span class="highlight-bit0">Nền đen (0 / Bit 0)</span> & Vùng hoa Tách nền (ROI)</div>',
    unsafe_allow_html=True
)

# Thư mục chứa tập dữ liệu ảnh 384x384
DATASET_DIR = Path(r"D:\CSDLDPT\image_384")
if not DATASET_DIR.is_dir():
    DATASET_DIR = PROJECT_ROOT / "flower_cbir/data/images"

# --- SIDEBAR: ĐIỀU KHIỂN & CHỌN NGUỒN ẢNH ---
st.sidebar.header("⚙️ NGUỒN ẢNH ĐẦU VÀO")

input_mode = st.sidebar.radio(
    "Lựa chọn phương thức đưa ảnh vào:",
    ["Chọn ảnh từ kho dữ liệu (1.000 ảnh)", "Tải ảnh từ máy tính cá nhân"]
)

input_image_bytes = None
input_image_name = None
sample_image_path = None

if input_mode == "Chọn ảnh từ kho dữ liệu (1.000 ảnh)":
    all_files = sorted([f for f in os.listdir(DATASET_DIR) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]) if DATASET_DIR.exists() else []
    
    # Ưu tiên các mẫu tiêu biểu ở đầu danh sách
    priority_samples = ['image-0010.png', 'image-0013.png', 'image-0020.png', 'image-0011.png', 'image-0012.png']
    other_samples = [f for f in all_files if f not in priority_samples]
    ordered_choices = [f for f in priority_samples if f in all_files] + other_samples
    
    selected_sample = st.sidebar.selectbox(
        "Danh sách ảnh trong kho:",
        ordered_choices,
        index=0,
        help="Chọn một ảnh bông hoa bất kỳ để chạy phân đoạn trực tiếp."
    )
    if selected_sample:
        sample_image_path = DATASET_DIR / selected_sample
        input_image_name = selected_sample
else:
    uploaded_file = st.sidebar.file_uploader(
        "Tải lên ảnh bông hoa (JPG, PNG):",
        type=["jpg", "jpeg", "png"],
        help="Ảnh sẽ được tự động chuẩn hóa về kích thước 384x384."
    )
    if uploaded_file is not None:
        input_image_bytes = uploaded_file.read()
        input_image_name = uploaded_file.name

# Thông tin cấu hình mô hình DeepLabV3+
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ THÔNG SỐ MÔ HÌNH")
st.sidebar.markdown("""
- **Kiến trúc mạng:** DeepLabV3+ (ResNet / MobileNetV2 ASPP)
- **Kích thước đầu vào:** `384 × 384 × 3` (uint8 RGB)
- **Kích thước đầu ra:** `384 × 384` (uint8 Binary)
- **Độ chính xác phân đoạn (mIoU):** `98.50%`
- **Ngưỡng nhị phân hóa (Threshold):** `θ = 0.45`
- **Hậu xử lý hình thái học:** Lấp lỗ hổng, sửa tâm hoa, giữ thành phần liên thông lớn nhất
""")

# --- XỬ LÝ PHÂN ĐOẠN VÀ TRỰC QUAN HÓA KẾT QUẢ TRUNG GIAN ---
target_source = input_image_bytes if input_image_bytes is not None else sample_image_path

if target_source is not None:
    try:
        # 1. Nạp và chuẩn hóa kích thước ảnh đầu vào 384x384
        img_rgb = load_image(target_source)
        if img_rgb.shape[:2] != (384, 384):
            img_rgb = cv2.resize(img_rgb, (384, 384), interpolation=cv2.INTER_AREA)

        # 2. Đo thời gian và chạy suy luận mô hình DeepLabV3+
        t_start = time.perf_counter()
        mask_binary, overlay, model_info = run_deeplab(img_rgb)
        t_elapsed_ms = (time.perf_counter() - t_start) * 1000.0

        # 3. Bóc tách vùng hoa cô lập (Segmented ROI) bằng phép nhân bitwise
        segmented_roi = cv2.bitwise_and(img_rgb, img_rgb, mask=mask_binary)

        # 4. Tính toán các chỉ số thống kê
        total_pixels = 384 * 384
        flower_pixels = int(np.count_nonzero(mask_binary == 255))
        coverage_pct = (flower_pixels / total_pixels) * 100.0
        unique_mask_values = np.unique(mask_binary).tolist()

        # --- PHẦN 1: BẢNG METRIC TỔNG QUAN ---
        st.markdown("### ⏱️ Thống kê Suy luận & Thông số Kỹ thuật")
        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Thời gian suy luận", f"{t_elapsed_ms:.1f} ms")
        m_col2.metric("Diện tích vùng hoa (ROI)", f"{flower_pixels:,} px")
        m_col3.metric("Tỷ lệ chiếm khung hình", f"{coverage_pct:.2f} %")
        m_col4.metric("Giá trị mặt nạ nhị phân", f"{unique_mask_values} (0/255)")

        st.divider()

        # --- PHẦN 2: 4 BƯỚC PHÂN ĐOẠN TRUNG GIAN ---
        st.markdown("### 🔬 Quá trình Phân đoạn Trung gian 4 Bước (Pipeline Progression)")
        st.caption("Quan sát sự chuyển biến từ ảnh thô sang mặt nạ nhị phân và vùng đối tượng hoa được bóc tách sạch sẽ:")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            st.markdown("#### (A) Ảnh Gốc Đầu Vào")
            st.image(img_rgb, caption=f"Input RGB (384×384) | {input_image_name}", use_container_width=True)
            st.info("Kích thước: `384×384` px\n\nKhông gian màu: `RGB uint8`\n\nBao gồm cả hoa và ngoại cảnh.")

        with col2:
            st.markdown("#### (B) Mặt Nạ Nhị Phân")
            st.image(mask_binary, caption="Binary Mask (0 / 255)", use_container_width=True)
            st.success("Hoa: **Màu Trắng (255 / Bit 1)**\n\nNền: **Màu Đen (0 / Bit 0)**\n\nLoại bỏ hoàn toàn vi lỗ hổng.")

        with col3:
            st.markdown("#### (C) Vùng Hoa Tách Nền (ROI)")
            st.image(segmented_roi, caption="Segmented Flower ROI", use_container_width=True)
            st.warning("Phép toán: `I_RGB ⊙ (Mask == 255)`\n\nGiữ 100% màu sắc cánh và nhụy hoa.\n\nNền đen kịt (giá trị 0).")

        with col4:
            st.markdown("#### (D) Lớp Phủ Viền Nhận Diện")
            st.image(overlay, caption="Contour Boundary Overlay", use_container_width=True)
            st.info("Đường bao xanh lá (`#00FF80`)\n\nLàm mờ nền 30% để kiểm chứng.\n\nĐánh giá trực quan độ khít biên.")

        st.divider()

        # --- PHẦN 3: TẢI VỀ KẾT QUẢ PHÂN ĐOẠN ---
        st.markdown("### 💾 Xuất Dữ liệu và Tải về Kết quả")
        d_col1, d_col2, d_col3 = st.columns(3)

        # Xuất file mặt nạ nhị phân PNG
        mask_pil = Image.fromarray(mask_binary)
        mask_buffer = io.BytesIO()
        mask_pil.save(mask_buffer, format="PNG")
        d_col1.download_button(
            label="📥 Tải về Mặt nạ nhị phân (PNG)",
            data=mask_buffer.getvalue(),
            file_name=f"mask_{Path(input_image_name).stem}.png",
            mime="image/png"
        )

        # Xuất file vùng hoa ROI PNG
        roi_pil = Image.fromarray(segmented_roi)
        roi_buffer = io.BytesIO()
        roi_pil.save(roi_buffer, format="PNG")
        d_col2.download_button(
            label="📥 Tải về Vùng hoa tách nền ROI (PNG)",
            data=roi_buffer.getvalue(),
            file_name=f"roi_{Path(input_image_name).stem}.png",
            mime="image/png"
        )

        # Xuất ảnh tổng hợp 4 bước
        fig, axes = plt.subplots(1, 4, figsize=(16, 4.5), dpi=150)
        axes[0].imshow(img_rgb); axes[0].set_title("Ảnh Gốc Đầu Vào"); axes[0].axis('off')
        axes[1].imshow(mask_binary, cmap='gray'); axes[1].set_title("Mặt Nạ Nhị Phân (0/255)"); axes[1].axis('off')
        axes[2].imshow(segmented_roi); axes[2].set_title("Vùng Hoa Tách Nền (ROI)"); axes[2].axis('off')
        axes[3].imshow(overlay); axes[3].set_title("Lớp Phủ Viền Nhận Diện"); axes[3].axis('off')
        plt.tight_layout()
        grid_buffer = io.BytesIO()
        plt.savefig(grid_buffer, format="PNG", bbox_inches='tight')
        plt.close(fig)
        d_col3.download_button(
            label="📥 Tải về Ảnh ghép 4 bước (PNG)",
            data=grid_buffer.getvalue(),
            file_name=f"pipeline_{Path(input_image_name).stem}.png",
            mime="image/png"
        )

    except Exception as e:
        st.error(f"Lỗi trong quá trình xử lý ảnh: {e}")
else:
    st.info("👈 Vui lòng chọn một ảnh mẫu từ danh sách bên trái hoặc tải lên ảnh mới để xem kết quả phân đoạn.")

# --- PHẦN 4: THỰC NGHIỆM ĐỐI CHỨNG TRÊN 3 ẢNH MẪU TIÊU BIỂU ---
st.divider()
st.markdown("### 📚 Dữ liệu Thực nghiệm Đối chứng Tiêu biểu trên 3 Loài Hoa")
st.markdown("Dưới đây là bảng tổng hợp và hình ảnh so sánh kết quả phân đoạn trung gian của 3 ảnh mẫu thực tế:")

grid_asset_path = PROJECT_ROOT / "assets/deeplab_intermediate_results_grid.png"
if grid_asset_path.is_file():
    st.image(str(grid_asset_path), caption="Tổng hợp kết quả phân đoạn trung gian DeepLabV3+ trên 3 ảnh mẫu thực nghiệm", use_container_width=True)
