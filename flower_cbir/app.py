import os
import sys
from pathlib import Path
import json
import time
import hashlib

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import streamlit as st
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image

from flower_cbir.core.preprocess import load_image
from flower_cbir.services.search import FlowerSearchEngine
from flower_cbir.services.database import FlowerDatabase
from flower_cbir.evaluation.evaluate import run_correctness_checks, run_ablation_study
from flower_cbir.settings import default_weights
from flower_cbir.core.contract import extraction_contract, fingerprint
plt.style.use('dark_background')


def safe_search(*args, **kwargs):
    try:
        return engine.search(*args, **kwargs)
    except (ValueError, RuntimeError, OSError, cv2.error) as error:
        st.error(f'Không thể xử lý ảnh: {error}. Vui lòng kiểm tra ảnh hoặc điều chỉnh khung bao bông hoa.')
        st.stop()


# --- Page Configuration & Custom Styling ---
st.set_page_config(
    page_title="CBIR - Hệ thống Tìm kiếm Ảnh Bông hoa",
    page_icon="🌸",
    layout="wide",
    initial_sidebar_state="expanded"
)

CUSTOM_CSS = """
<style>
    /* Global styling */
    .main-header {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #FF4B4B 0%, #FF8533 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.2rem;
    }
    .sub-header {
        color: #718096;
        font-size: 1.05rem;
        margin-bottom: 1.5rem;
    }
    .card-box {
        background: rgba(255, 255, 255, 0.05);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 12px;
        padding: 16px;
        margin-bottom: 14px;
    }
    .metric-badge {
        display: inline-block;
        padding: 4px 10px;
        border-radius: 16px;
        font-weight: 600;
        font-size: 0.85rem;
    }
    .badge-rank-1 { background-color: #ffd700; color: #000; }
    .badge-rank-2 { background-color: #c0c0c0; color: #000; }
    .badge-rank-3 { background-color: #cd7f32; color: #fff; }
    .badge-rank-other { background-color: #4a5568; color: #fff; }
    .stTabs [data-baseweb="tab-list"] {
        gap: 12px;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        font-weight: 600;
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)


@st.cache_resource
def get_search_engine(active_version_id, contract_fingerprint):
    """Cache the search engine and index in memory for instant query responses."""
    db = FlowerDatabase()
    try:
        engine = FlowerSearchEngine(db.load_index())
    except Exception:
        db.close()
        raise
    return engine, db


# Load engine and database
try:
    probe = FlowerDatabase()
    try:
        active_version_id=probe.active_version()
    finally:probe.close()
    engine, db = get_search_engine(active_version_id,fingerprint(extraction_contract()))
except Exception as e:
    st.error(f"Lỗi khởi tạo hệ thống tìm kiếm: {e}")
    st.stop()


# --- Sidebar Navigation & Settings ---
st.sidebar.caption(f'Index: {engine.version_id}')
st.sidebar.markdown("### 🌸 CBIR Flower Retrieval")
st.sidebar.markdown(f"**Kho dữ liệu:** {engine.num_images} ảnh bông hoa (163 chiều)")

st.sidebar.divider()
st.sidebar.markdown("#### ⚙️ Cấu hình Trọng số Đặc trưng")
configured_weights = default_weights()
w1 = st.sidebar.slider("w₁ - F1 Màu sắc tổng thể (78D)", 0.0, 1.0, configured_weights[0], 0.05)
w2 = st.sidebar.slider("w₂ - F2 Màu sắc theo vị trí (18D)", 0.0, 1.0, configured_weights[1], 0.05)
w3 = st.sidebar.slider("w₃ - F3 Kết cấu bề mặt LBP (28D)", 0.0, 1.0, configured_weights[2], 0.05)
w4 = st.sidebar.slider("w₄ - F4 Hình dạng & Cấu trúc (39D)", 0.0, 1.0, configured_weights[3], 0.05)

w_sum = w1 + w2 + w3 + w4
if w_sum > 0:
    weights = (w1 / w_sum, w2 / w_sum, w3 / w_sum, w4 / w_sum)
else:
    weights = (0.25, 0.25, 0.25, 0.25)

st.sidebar.caption(f"Trọng số chuẩn hóa: F1={weights[0]:.2f}, F2={weights[1]:.2f}, F3={weights[2]:.2f}, F4={weights[3]:.2f}")

st.sidebar.divider()
top_k = st.sidebar.slider("Số lượng kết quả (Top K)", 1, 10, 5)
leave_one_out = st.sidebar.checkbox(
    "Leave-One-Out (loại bỏ ảnh trùng)", 
    value=False,
    help="Loại chính ảnh đã được xác minh theo nội dung pixel, không dựa vào tên file. Điểm tương đồng không phải xác suất đúng loài."
)


# --- Main Application Header ---
st.markdown('<div class="main-header">HỆ THỐNG TRUY VẤN ẢNH BÔNG HOA THEO NỘI DUNG (CBIR)</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Môn học: Hệ Cơ sở Dữ liệu Đa phương tiện | Vector đặc trưng 163 chiều chuẩn hóa chọn lọc</div>', unsafe_allow_html=True)

tab_search, tab_explain, tab_eval, tab_history = st.tabs([
    "🔍 Tìm kiếm Ảnh (Search)",
    "📊 Chế độ Giải thích (Explainability)",
    "🧪 Đánh giá & Ablation Study",
    "📜 Lịch sử Truy vấn"
])


# --- Helper Function: Plotting Explanation Figures ---
def plot_f1_visuals(exp_f1):
    hs_2d = exp_f1["hs_2d"]
    v_hist = exp_f1["v_hist"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.5), facecolor='#17212f')
    # HS 2D Heatmap
    im = axes[0].imshow(hs_2d.T, cmap="viridis", aspect="auto", origin="lower")
    axes[0].set_title("HSV: Heatmap phân bố H-S (12 x 6 bins)", fontsize=10, color="white")
    axes[0].set_xlabel("Hue Bin (0..11, mỗi 30°)", color="white")
    axes[0].set_ylabel("Sat Bin (0..5)", color="white")
    axes[0].tick_params(colors="white")
    plt.colorbar(im, ax=axes[0])

    # V 1D Histogram
    axes[1].bar(range(1, 7), v_hist, color="#FFA500", edgecolor="white", alpha=0.85)
    axes[1].set_title("Độ sáng V Histogram (6 bins)", fontsize=10, color="white")
    axes[1].set_xlabel("V Bin (0..1)", color="white")
    axes[1].set_ylabel("Tần suất chuẩn hóa", color="white")
    axes[1].tick_params(colors="white")
    axes[1].set_ylim(0, max(0.1, np.max(v_hist) * 1.2))

    fig.tight_layout()
    return fig


def plot_f3_visuals(exp_f3):
    lbp8 = exp_f3["lbp8_hist"]
    lbp16 = exp_f3["lbp16_hist"]

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.2), facecolor='#17212f')
    axes[0].bar(range(len(lbp8)), lbp8, color="#00CED1", edgecolor="white", alpha=0.85)
    axes[0].set_title("LBP riu2 (P=8, R=1) - 10 Bins", fontsize=10, color="white")
    axes[0].set_xlabel("Mã LBP (0..9)", color="white")
    axes[0].set_ylabel("Tần suất", color="white")
    axes[0].tick_params(colors="white")

    axes[1].bar(range(len(lbp16)), lbp16, color="#9370DB", edgecolor="white", alpha=0.85)
    axes[1].set_title("LBP riu2 (P=16, R=2) - 18 Bins", fontsize=10, color="white")
    axes[1].set_xlabel("Mã LBP (0..17)", color="white")
    axes[1].set_ylabel("Tần suất", color="white")
    axes[1].tick_params(colors="white")

    fig.tight_layout()
    return fig


def plot_f4_visuals(exp_f4):
    r_signal = exp_f4["r_signal"]
    spectrum = exp_f4["radial_spectrum"]
    iap = exp_f4["iap_matrix"]
    d_vals = exp_f4["d_vals"]

    fig, axes = plt.subplots(2, 2, figsize=(10, 6.5), facecolor='#17212f')

    # 1. Radial distance signal r[n]
    axes[0, 0].plot(r_signal, color="#FF69B4", lw=1.8)
    axes[0, 0].set_title("Biên tia r[n] (128 góc quay)", fontsize=10, color="white")
    axes[0, 0].set_xlabel("Góc n (0..127)", color="white")
    axes[0, 0].set_ylabel("Bán kính r (pixels)", color="white")
    axes[0, 0].tick_params(colors="white")

    # 2. Fourier radial spectrum
    axes[0, 1].stem(range(1, 13), spectrum, linefmt="C1-", markerfmt="C1o", basefmt="gray")
    axes[0, 1].set_title("Phổ Fourier Radial S[1]..S[12]", fontsize=10, color="white")
    axes[0, 1].set_xlabel("Thành phần tần số k", color="white")
    axes[0, 1].set_ylabel("|R[k]| / |R[0]|", color="white")
    axes[0, 1].tick_params(colors="white")

    # 3. IAP Matrix (4 rings x 4 bins)
    im = axes[1, 0].imshow(iap, cmap="plasma", aspect="auto")
    axes[1, 0].set_title("IAP Matrix: 4 vành x 4 hướng góc", fontsize=10, color="white")
    axes[1, 0].set_xlabel("Hướng góc m (0=vòng, 2=tia)", color="white")
    axes[1, 0].set_ylabel("Vành bán kính (0=trong..3=ngoài)", color="white")
    axes[1, 0].tick_params(colors="white")
    plt.colorbar(im, ax=axes[1, 0])

    # 4. IEP Edge density across 5 rings
    axes[1, 1].plot(range(1, 6), d_vals, marker='o', color="#32CD32", lw=2)
    axes[1, 1].set_title(f"Mật độ cạnh IEP 5 vành (Sedge={exp_f4['s_edge']:.3f}, CVedge={exp_f4['cv_edge']:.3f})", fontsize=9, color="white")
    axes[1, 1].set_xlabel("Vành IEP (1=trong..5=ngoài)", color="white")
    axes[1, 1].set_ylabel("Mật độ cạnh dk", color="white")
    axes[1, 1].tick_params(colors="white")

    fig.tight_layout()
    return fig


# --- TAB 1: SEARCH ---
with tab_search:
    col_input, col_config = st.columns([1.2, 1])

    with col_input:
        st.markdown("##### 📤 Tải ảnh truy vấn hoặc Chọn ảnh mẫu")
        query_option = st.radio("Nguồn ảnh:", ["Tải file từ máy tính", "Chọn mẫu từ kho dữ liệu"], horizontal=True)

        query_source = None
        query_filename = None
        preset_mask = None

        if query_option == "Tải file từ máy tính":
            uploaded_file = st.file_uploader("Chọn ảnh bông hoa (JPG, PNG):", type=["jpg", "jpeg", "png"])
            if uploaded_file is not None:
                query_source = uploaded_file.read()
                query_filename = uploaded_file.name
                matched = engine.match_source(query_source)
                if matched:
                    source_rgb = load_image(query_source)
                    if source_rgb.shape[:2] == (384, 384):
                        preset_mask = cv2.imread(matched['mask_path'], cv2.IMREAD_GRAYSCALE)
                    st.success(f"Nội dung ảnh khớp ID #{matched['image_id']} trong kho.")
        else:
            sample_choices = [m["filename"] for m in engine.metadata]
            chosen_file = st.selectbox("Chọn ảnh trong kho (1.000 ảnh):", sample_choices, index=0)
            match_meta = next(m for m in engine.metadata if m["filename"] == chosen_file)
            query_source = match_meta["image_path"]
            query_filename = match_meta["filename"]
            preset_mask = cv2.imread(match_meta["mask_path"], cv2.IMREAD_GRAYSCALE)

    with col_config:
        st.markdown("##### 🌸 Phân đoạn DeepLabV3+ 384 v2")
        st.caption('Ảnh đã chuẩn 384×384 được giữ nguyên. Ảnh kích thước khác được định vị hoa, cắt vuông và thêm lề 10%.')
        enable_custom_box = st.checkbox("Điều chỉnh khung bao bông hoa", value=False)
        if enable_custom_box:
            source_shape=load_image(query_source).shape[:2] if query_source is not None else (384,384)
            ih,iw=source_shape
            c1, c2 = st.columns(2)
            rx = c1.number_input("Tọa độ X",0,max(0,iw-1),0)
            ry = c1.number_input("Tọa độ Y",0,max(0,ih-1),0)
            rw = c2.number_input("Chiều rộng hoa",1,iw,iw)
            rh = c2.number_input("Chiều cao hoa",1,ih,ih)
            flower_box=(rx,ry,rw,rh)
        else:
            flower_box=None

        use_stored_mask = st.checkbox(
            "Dùng mask DeepLabV3+ đã lưu cho ảnh trong kho",
            value=(preset_mask is not None and not enable_custom_box),disabled=enable_custom_box,
            help="Mask đã lưu được tạo bằng DeepLabV3+ 384 v2 và cùng hậu xử lý. Bỏ chọn để chạy lại mô hình trên ảnh này."
        )

    st.divider()

    if query_source is not None:
        source_identity = hashlib.sha256(query_source if isinstance(query_source, bytes) else str(query_source).encode()).hexdigest()
        extraction_identity = (engine.version_id,source_identity,bool(use_stored_mask and preset_mask is not None and not enable_custom_box),flower_box)
        if st.session_state.get('extraction_identity') != extraction_identity:
            st.session_state.pop('last_query_result', None)
        elif 'last_query_result' in st.session_state:
            previous = st.session_state['last_query_result']
            if tuple(previous['weights']) != tuple(weights) or st.session_state.get('ranking_settings') != (top_k, leave_one_out):
                st.session_state['last_query_result'] = engine.rerank(previous, weights, top_k, leave_one_out)
                st.session_state['ranking_settings'] = (top_k, leave_one_out)
        btn_search = st.button("🚀 BẮT ĐẦU TRUY VẤN ẢNH TƯƠNG ĐỒNG", type="primary", width='stretch')

        if btn_search or "last_query_result" in st.session_state:
            if btn_search:
                with st.spinner("Đang thực hiện tiền xử lý, phân đoạn, trích xuất 163 chiều và tìm kiếm..."):
                    mask_to_use = preset_mask if (use_stored_mask and preset_mask is not None and not enable_custom_box) else None
                    search_res = safe_search(
                        query_source,
                        query_mask=mask_to_use,
                        flower_box=flower_box,
                        weights=weights,
                        top_k=top_k,
                        leave_one_out=leave_one_out,
                        query_filename=query_filename,
                        use_gallery_mask=bool(use_stored_mask and not enable_custom_box)
                    )
                    st.session_state["last_query_result"] = search_res
                    st.session_state["query_filename"] = query_filename or "uploaded_image.png"
                    st.session_state['extraction_identity'] = extraction_identity
                    st.session_state['ranking_settings'] = (top_k, leave_one_out)

                    # Log to database
                    db.log_retrieval(
                        query_filename=st.session_state["query_filename"],
                        query_norm_vec=search_res["query_info"]["norm_vector"],
                        top5_results=search_res["results"],
                        weights=weights,
                        execution_ms=search_res["timings"]["total_backend_ms"],
                        version_id=engine.version_id,
                        query_context={'stored_mask':mask_to_use is not None,'flower_box':flower_box,'top_k':top_k,
                                       'preparation':search_res['query_info']['preparation'],
                                       'segmentation':search_res['query_info']['segmentation'],'leave_one_out':leave_one_out}
                    )

            res_data = st.session_state["last_query_result"]
            q_info = res_data["query_info"]
            results = res_data["results"]
            timings = res_data["timings"]

            # Display Query Image & Timing summary
            st.markdown("### 📌 Kết quả Tìm kiếm")

            col_q, col_summary = st.columns([1, 2.5])
            with col_q:
                st.markdown("**Ảnh truy vấn (384x384 + Mask):**")
                st.image(q_info["overlay"], caption=f"Query: {st.session_state.get('query_filename', '')}", width='stretch')

            with col_summary:
                st.markdown("**⏱️ Thời gian xử lý Backend:**")
                t_col1, t_col2, t_col3, t_col4, t_col5 = st.columns(5)
                t_col1.metric("Tiền xử lý", f"{timings['preprocess_ms']:.1f} ms")
                t_col2.metric("Phân đoạn", f"{timings['segmentation_ms']:.1f} ms")
                t_col3.metric("Trích xuất 163D", f"{timings['extraction_ms']:.1f} ms")
                t_col4.metric("Tìm kiếm", f"{timings['search_ms']:.2f} ms")
                t_col5.metric("Tổng thời gian", f"{timings['total_backend_ms']:.1f} ms")

                st.info(f"💡 Hệ thống quét chính xác toàn bộ **{engine.num_images}** vector trong kho bằng NumPy. Độ đo khoảng cách kết hợp Hellinger và RMS chuẩn hóa theo 4 nhóm thuộc tính.")

            st.markdown(f"#### 🏆 Top {len(results)} Ảnh Tương đồng Nhất:")
            if res_data.get('reused_query_features'):
                st.caption('Đã dùng lại vector truy vấn; thời gian bên dưới chỉ đo xếp hạng lại.')
            cols_top = st.columns(len(results))
            for i, item in enumerate(results):
                with cols_top[i]:
                    badge_class = f"badge-rank-{item['rank']}" if item['rank'] <= 3 else "badge-rank-other"
                    st.markdown(f'<span class="metric-badge {badge_class}">RANK #{item["rank"]}</span>', unsafe_allow_html=True)
                    if os.path.exists(item["image_path"]):
                        st.image(item["image_path"], width='stretch')
                    else:
                        st.warning(f"File {item['filename']} không khả dụng")

                    st.markdown(f"**{item['filename']}**")
                    st.markdown(f"Độ tương đồng: **{item['similarity']*100:.1f}%**")
                    st.caption(f"Khoảng cách D: `{item['distance']:.4f}`")

                    # Expandable breakdown
                    with st.expander("Chi tiết đóng góp"):
                        st.write(f"- D₁ (Màu sắc): `{item['D1']:.4f}`")
                        st.write(f"- D₂ (Màu vùng): `{item['D2']:.4f}`")
                        st.write(f"- D₃ (Kết cấu): `{item['D3']:.4f}`")
                        st.write(f"- D₄ (Hình dạng): `{item['D4']:.4f}`")
                        st.write(f"  • Scalars: `{item['f4_details']['scalars']:.3f}`")
                        st.write(f"  • Fourier: `{item['f4_details']['fourier']:.3f}`")
                        st.write(f"  • IAP: `{item['f4_details']['iap']:.3f}`")
                        st.write(f"  • IEP: `{item['f4_details']['iep']:.3f}`")


# --- TAB 2: EXPLAINABILITY ---
with tab_explain:
    st.markdown("### 🔬 Chế độ Giải thích Trực quan Đặc trưng (Explainability Mode)")
    st.markdown("Toàn bộ các tín hiệu hình học, biểu đồ màu, kết cấu và hình dáng phục vụ giải thích tính toán tương đồng:")

    if "last_query_result" in st.session_state:
        res_data = st.session_state["last_query_result"]
        q_info = res_data["query_info"]
        geom = q_info["explanation"]["geometry"]
        f1_exp = q_info["explanation"]["f1"]
        f2_exp = q_info["explanation"]["f2"]
        f3_exp = q_info["explanation"]["f3"]
        f4_exp = q_info["explanation"]["f4"]

        # Section 1: Visual Masks and Geometry
        st.markdown("#### 1. Hình học và Phân đoạn Vùng Hoa (Geometry & Masks)")
        g_col1, g_col2, g_col3, g_col4 = st.columns(4)
        g_col1.image(q_info["image_rgb"], caption="Ảnh RGB 384x384", width='stretch')
        g_col2.image(q_info["mask_binary"], caption="Mặt nạ M (Nhị phân 0/255)", width='stretch')
        g_col3.image(geom["m_int_binary"], caption="Mặt nạ nội bộ M_int (Erode B5)", width='stretch')

        # Draw rings on image for visualization
        ring_vis = q_info["image_rgb"].copy()
        xc, yc = int(round(geom["centroid"][0])), int(round(geom["centroid"][1]))
        cv2.circle(ring_vis, (xc, yc), 4, (0, 0, 255), -1)
        # Draw concentric circles corresponding to rho = 1/3, 2/3, 1.0
        r_max = int(round(geom["r_max"]))
        for r_factor, color in [(1/3, (255, 255, 0)), (2/3, (0, 255, 255)), (1.0, (255, 0, 255))]:
            cv2.circle(ring_vis, (xc, yc), int(r_max * r_factor), color, 1)
        g_col4.image(ring_vis, caption=f"Tâm ({xc},{yc}) + 3 Vành Lab", width='stretch')

        st.caption(f"Diện tích A: **{int(geom['A'])} px** | Diện tích M_int: **{int(geom['A_int'])} px** | Chu vi P: **{geom['perimeter']:.1f} px** | Bán kính r_max: **{geom['r_max']:.1f} px**")

        st.divider()

        # Section 2: F1 and F2 (Color)
        st.markdown("#### 2. Đặc trưng Màu sắc (F1 & F2 - 96 chiều)")
        col_f1_plot, col_f2_table = st.columns([1.4, 1])

        with col_f1_plot:
            fig_f1 = plot_f1_visuals(f1_exp)
            st.pyplot(fig_f1)
            plt.close(fig_f1)

        with col_f2_table:
            st.markdown("**F2: Thống kê Lab theo 3 vành đồng tâm:**")
            ring_rows = []
            for r in f2_exp["ring_stats"]:
                ring_rows.append({
                    "Vành": f"Vành {r['ring']+1}",
                    "Pixels": r["count"],
                    "μL / σL": f"{r['mu_L']:.1f} / {r['std_L']:.1f}",
                    "μa / σa": f"{r['mu_a']:.1f} / {r['std_a']:.1f}",
                    "μb / σb": f"{r['mu_b']:.1f} / {r['std_b']:.1f}",
                })
            st.dataframe(ring_rows, width='stretch')

        st.divider()

        # Section 3: F3 (Texture)
        st.markdown("#### 3. Đặc trưng Kết cấu Bề mặt Multi-Radius LBP (F3 - 28 chiều)")
        fig_f3 = plot_f3_visuals(f3_exp)
        st.pyplot(fig_f3)
        plt.close(fig_f3)

        st.divider()

        # Section 4: F4 (Shape & Internal Structure)
        st.markdown("#### 4. Đặc trưng Hình dạng & Cấu trúc Nội bộ (F4 - 39 chiều)")
        sc = f4_exp["scalars"]
        s_col1, s_col2, s_col3, s_col4 = st.columns(4)
        s_col1.metric("F4.1 Độ tròn (Circularity)", f"{sc['circularity']:.4f}")
        s_col2.metric("F4.2 Độ đặc (Solidity)", f"{sc['solidity']:.4f}")
        s_col3.metric("F4.3 Tỷ lệ trục (Aspect Ratio)", f"{sc['aspect_ratio']:.4f}")
        s_col4.metric("F4.4 Lấp đầy hộp (Rotated Extent)", f"{sc['rotated_extent']:.4f}")

        fig_f4 = plot_f4_visuals(f4_exp)
        st.pyplot(fig_f4)
        plt.close(fig_f4)

        st.divider()

        # Section 5: Feature Vector Values Inspection
        st.markdown("#### 5. Chi tiết Vector Đặc trưng 163 Chiều")
        vec_col1, vec_col2 = st.columns(2)
        with vec_col1:
            st.markdown("**Vector Thô (Raw Vector 163D):**")
            st.text_area("Giá trị float32:", value=str(q_info["raw_vector"].tolist()), height=150)
        with vec_col2:
            st.markdown("**Vector Sau Chuẩn hóa Chọn lọc (Search Vector 163D):**")
            st.text_area("Z-score F2/F4, giữ nguyên F1/F3:", value=str(q_info["norm_vector"].tolist()), height=150)

    else:
        st.info("💡 Vui lòng thực hiện một lượt tìm kiếm ở tab **Tìm kiếm Ảnh** để xem các biểu đồ giải thích trực quan.")


# --- TAB 3: EVALUATION & ABLATION STUDY ---
with tab_eval:
    st.markdown("### 🧪 Kiểm tra Tính Đúng đắn & Nghiên cứu Loại bỏ (Ablation Study)")

    col_chk, col_abl = st.columns(2)

    with col_chk:
        st.markdown("#### 1. Kiểm tra Bất biến Toán học (Mathematical Invariants)")
        if st.button("Chạy Kiểm tra Toàn diện", key="btn_run_tests"):
            with st.spinner("Đang kiểm tra ma trận 1000 ảnh..."):
                report = run_correctness_checks(engine)
                for t in report["tests"]:
                    if t["passed"]:
                        st.success(f"✅ {t['name']}: {t['detail']}")
                    else:
                        st.error(f"❌ {t['name']}: {t['detail']}")

    with col_abl:
        st.markdown("#### 2. Thử nghiệm Loại bỏ Thành phần (Ablation)")
        ablate_idx = st.number_input("Chọn chỉ số ảnh truy vấn trong kho (0..999):", 0, engine.num_images - 1, 0)
        if st.button("Chạy Ablation Study", key="btn_run_ablation"):
            with st.spinner("Đang tính toán các cấu hình..."):
                abl_res = run_ablation_study(engine, query_idx=ablate_idx)
                st.markdown(f"**Kết quả Ablation cho ảnh: `{engine.metadata[ablate_idx]['filename']}`**")

                summary_table = []
                for name, data in abl_res.items():
                    top1 = data["top5"][0]["filename"]
                    sim1 = data["top5"][0]["similarity"]
                    summary_table.append({
                        "Cấu hình": name,
                        "Top-1 Tương đồng": top1,
                        "Điểm S(Top 1)": f"{sim1*100:.1f}%",
                        "Thời gian": f"{data['search_ms']:.2f} ms"
                    })
                st.dataframe(summary_table, width='stretch')


# --- TAB 4: RETRIEVAL HISTORY ---
with tab_history:
    st.markdown("### 📜 Lịch sử Truy vấn & Quản trị CSDL")
    history_records = db.get_history(limit=15)

    if len(history_records) > 0:
        h_table = []
        for r in history_records:
            h_table.append({
                "ID": r["query_id"],
                "File ảnh": r["query_filename"],
                "Top-1 Kết quả": r["top5"][0]["filename"] if len(r["top5"]) > 0 else "N/A",
                "Độ tương đồng": f"{r['top5'][0]['similarity']*100:.1f}%" if len(r["top5"]) > 0 else "N/A",
                "Thời gian": f"{r['execution_ms']:.1f} ms",
                "Phiên bản": r['version_id'] or 'Lịch sử cũ, chưa ghi phiên bản',
                "Trọng số": str(r['weights']) if r['weights'] is not None else 'Chưa ghi',
                "Thời điểm": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(r["timestamp"]))
            })
        st.dataframe(h_table, width='stretch')
    else:
        st.info("Chưa có lịch sử truy vấn nào được lưu.")
