# BÁO CÁO KỸ THUẬT: THIẾT KẾ CƠ SỞ DỮ LIỆU ĐA PHƯƠNG TIỆN VÀ HỆ THỐNG TRUY HỒI ẢNH HOA (CBIR)

**Học phần:** Hệ Cơ sở Dữ liệu Đa phương tiện  
**Đề tài:** Xây dựng hệ thống lưu trữ và tìm kiếm ảnh bông hoa dựa trên nội dung (Content-Based Image Retrieval - CBIR)  
**Quy mô dữ liệu:** 1.000 ảnh hoa kích thước chuẩn hóa $384 \times 384$ pixels  
**Không gian đặc trưng:** 163 chiều ($\mathbf{F}_1 \to \mathbf{F}_4$)  

---

## 3.1. KIẾN TRÚC TỔNG THỂ HỆ THỐNG (SYSTEM ARCHITECTURE)

Hệ thống truy hồi ảnh hoa dựa trên nội dung được thiết kế theo **Mô hình kiến trúc lai hai tầng (Hybrid Two-Tier Architecture)**, kết hợp giữa tầng lưu trữ dữ liệu quan hệ bền vững và tầng tính toán so khớp véc-tơ hóa hiệu năng cao trong bộ nhớ chính:

```text
┌────────────────────────────────────────────────────────────────────────┐
│                   GIAO DIỆN NGƯỜI DÙNG & TÌM KIẾM                       │
│              (Streamlit Web Application / Interactive UI)              │
│       - Tiếp nhận ảnh truy vấn mới / Lựa chọn ảnh từ kho CSDL          │
│       - Điều chỉnh thanh trượt trọng số thời gian thực [w1, w2, w3, w4]│
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Ảnh đầu vào RGB 384×384)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               TIỀN XỬ LÝ & TRÍCH XUẤT ĐẶC TRƯNG TRUY VẤN               │
│  1. Phân đoạn ngữ nghĩa tự động DeepLabV3+ (mIoU 98.50%) → Mask nhị phân│
│  2. Hậu xử lý hình thái học (Fill holes, Keep nearby, Repair center)   │
│  3. Trích xuất véc-tơ 163 chiều thô theo Bảng 2.1                      │
│  4. Chuẩn hóa Z-score chọn lọc (Sử dụng tham số μ, σ cố định từ CSDL)  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │ (Véc-tơ truy vấn q_norm ∈ ℝ¹⁶³)
                                    ▼
┌────────────────────────────────────────────────────────────────────────┐
│               TẦNG TRUY HỒI TRONG BỘ NHỚ (IN-MEMORY ENGINE)            │
│        - Tải trước ma trận véc-tơ chuẩn hóa N × 163D từ CSDL lên RAM   │
│        - Tính toán song song ma trận (NumPy SIMD Vectorization)        │
│        - Đo khoảng cách khối: Hellinger (F1, F3) & Transformed RMS (F2, F4)
│        - Tổng hợp tuyến tính có trọng số: D = ∑ wi·Di                  │
│        - Sắp xếp và xếp hạng Top-K (< 1.5 ms cho 1.000 ảnh)            │
└───────────────────────────────────▲────────────────────────────────────┘
                                    │ Đọc Index / Xác thực phiên bản
                                    │ (Snapshot nhất quán)
┌───────────────────────────────────┴────────────────────────────────────┐
│              TẦNG LƯU TRỮ BỀN VỮNG (PERSISTENT RDBMS)                  │
│                     PostgreSQL Database Server                         │
│  - flower_images: Siêu dữ liệu, đường dẫn, mã băm SHA-256 pixel        │
│  - flower_features: Lưu trữ 163 chiều thô phân rã theo 12 thuộc tính   │
│  - feature_normalization: Lưu trữ tham số thống kê chuẩn hóa μ, σ      │
│  - normalized_features: Véc-tơ 163 chiều đã chuẩn hóa toàn diện        │
│  - cbir_index_versions & cbir_index_entries: Quản lý phiên bản chỉ mục │
└────────────────────────────────────────────────────────────────────────┘
```

Mô hình kiến trúc phân tách rõ ràng hai chức năng cốt lõi:
1. **Tầng Lưu trữ Bền vững (Persistent Storage Layer):** Đảm nhiệm vai trò quản lý siêu dữ liệu, bảo đảm tính toàn vẹn tham chiếu, kiểm soát cấu trúc dữ liệu chặt chẽ và phiên bản hóa chỉ mục trích rút.
2. **Tầng Truy hồi Trong bộ nhớ (In-Memory Retrieval Engine):** Đảm nhiệm việc thực thi các phép toán đại số tuyến tính trên ma trận dữ liệu đã nạp sẵn vào RAM, bảo đảm độ trễ đáp ứng truy vấn ở mức phần nghìn giây ($\approx 1.2\text{ ms}$).

---

## 3.2. THIẾT KẾ VÀ QUẢN TRỊ CƠ SỞ DỮ LIỆU POSTGRESQL

### 3.2.1. Lý do lựa chọn Hệ quản trị CSDL Quan hệ PostgreSQL

Việc lựa chọn PostgreSQL cho hệ thống quản trị cơ sở dữ liệu đa phương tiện xuất phát từ các yêu cầu kỹ thuật nền tảng:

1. **Bảo đảm tính toàn vẹn và nhất quán dữ liệu (ACID Guarantees):**  
   Trong hệ thống CBIR, mỗi đối tượng ảnh bao gồm ba thực thể gắn kết mật thiết: tệp ảnh gốc, mặt nạ phân đoạn nhị phân và véc-tơ đặc trưng 163 chiều. PostgreSQL cung cấp cơ chế khóa ngoại nghiêm ngặt (`FOREIGN KEY ... REFERENCES ... ON DELETE CASCADE`), loại bỏ hoàn toàn nguy cơ xuất hiện các véc-tơ mồ côi hoặc sai lệch tham chiếu khi cập nhật hoặc xóa dữ liệu.

2. **Hỗ trợ kiểu dữ liệu mảng số thực đa chiều nguyên bản (`DOUBLE PRECISION[]`):**  
   PostgreSQL hỗ trợ kiểu dữ liệu mảng bản địa cho phép lưu trữ trực tiếp các véc-tơ số thực dấu phẩy động 64-bit với kích thước cố định. Điều này loại bỏ hoàn toàn chi phí tuần tự hóa/giải tuần tự hóa (serialization overhead) so với việc lưu trữ qua chuỗi JSON hay Text thông thường.

3. **Cơ chế ràng buộc toàn vẹn mảng chặt chẽ (`CHECK Constraints`):**  
   PostgreSQL cho phép áp dụng các hàm kiểm tra cấu trúc mảng trực tiếp trong định nghĩa bảng:
   - `array_ndims(cột) = 1`: Đảm bảo véc-tơ luôn là mảng một chiều.
   - `array_length(cột, 1) = D`: Đảm bảo độ dài véc-tơ khớp chính xác tuyệt đối với số chiều lý thuyết quy định ($D \in \{72, 6, 9, 28, 12, 16, 7, 163\}$).
   - `array_position(cột, NULL) IS NULL`: Ngăn chặn triệt để sự xuất hiện của các phần tử khuyết thiếu (`NULL`) trong véc-tơ đặc trưng.

4. **Quản lý phiên bản trích rút bất biến (Immutable Index Versioning):**  
   Mô hình dữ liệu thiết kế bảng quản lý hợp đồng phiên bản (`cbir_index_versions`), cho phép đánh dấu phiên bản trích rút hợp lệ, lưu vết cấu hình mô hình phân đoạn và tham số trích rút tương ứng. Khi cần thử nghiệm mô hình phân đoạn mới hoặc nâng cấp tập đặc trưng, dữ liệu cũ vẫn được bảo toàn nguyên vẹn.

---

### 3.2.2. Lược đồ Cơ sở Dữ liệu Chi tiết (Database Schema)

Hệ thống phân rã cấu trúc dữ liệu thành 6 bảng quan hệ chuẩn hóa thuộc schema quản lý `flower_ir`:

#### 1. Bảng quản lý ảnh và mặt nạ (`flower_images`)
Lưu trữ định danh ảnh, đường dẫn tệp tin, kích thước hình học và các giá trị mã băm SHA-256 nhằm phát hiện trùng lặp mức độ bit:

```sql
CREATE SCHEMA IF NOT EXISTS flower_ir;

CREATE TABLE IF NOT EXISTS flower_ir.flower_images (
    image_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    image_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    image_sha256 CHAR(64) NOT NULL,
    mask_sha256 CHAR(64) NOT NULL,
    pixel_sha256 CHAR(64) NOT NULL,
    width INTEGER NOT NULL CHECK (width = 384),
    height INTEGER NOT NULL CHECK (height = 384),
    flower_pixels INTEGER NOT NULL CHECK (flower_pixels > 0 AND flower_pixels < 384 * 384),
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_images_sha256 ON flower_ir.flower_images(image_sha256);
CREATE INDEX IF NOT EXISTS idx_images_pixel_sha ON flower_ir.flower_images(pixel_sha256);
```

#### 2. Bảng lưu trữ đặc trưng thô Tầng 1 (`flower_features`)
Lưu trữ 12 thuộc tính phân rã theo 4 nhóm lớn (tổng cộng 163 chiều) theo đúng dạng đo lường vật lý ban đầu:

```sql
CREATE TABLE IF NOT EXISTS flower_ir.flower_features (
    image_id BIGINT PRIMARY KEY REFERENCES flower_ir.flower_images(image_id) ON DELETE CASCADE,
    hs_histogram DOUBLE PRECISION[] NOT NULL,
    v_histogram DOUBLE PRECISION[] NOT NULL,
    lab_mean DOUBLE PRECISION[] NOT NULL,
    lab_std DOUBLE PRECISION[] NOT NULL,
    lbp_histogram DOUBLE PRECISION[] NOT NULL,
    circularity DOUBLE PRECISION NOT NULL,
    solidity DOUBLE PRECISION NOT NULL,
    aspect_ratio DOUBLE PRECISION NOT NULL,
    rotated_extent DOUBLE PRECISION NOT NULL,
    radial_spectrum DOUBLE PRECISION[] NOT NULL,
    iap DOUBLE PRECISION[] NOT NULL,
    iep DOUBLE PRECISION[] NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_hs_72 CHECK (array_ndims(hs_histogram) = 1 AND array_length(hs_histogram, 1) = 72 AND array_position(hs_histogram, NULL) IS NULL),
    CONSTRAINT check_v_6 CHECK (array_ndims(v_histogram) = 1 AND array_length(v_histogram, 1) = 6 AND array_position(v_histogram, NULL) IS NULL),
    CONSTRAINT check_lab_mean_9 CHECK (array_ndims(lab_mean) = 1 AND array_length(lab_mean, 1) = 9 AND array_position(lab_mean, NULL) IS NULL),
    CONSTRAINT check_lab_std_9 CHECK (array_ndims(lab_std) = 1 AND array_length(lab_std, 1) = 9 AND array_position(lab_std, NULL) IS NULL),
    CONSTRAINT check_lbp_28 CHECK (array_ndims(lbp_histogram) = 1 AND array_length(lbp_histogram, 1) = 28 AND array_position(lbp_histogram, NULL) IS NULL),
    CONSTRAINT check_fourier_12 CHECK (array_ndims(radial_spectrum) = 1 AND array_length(radial_spectrum, 1) = 12 AND array_position(radial_spectrum, NULL) IS NULL),
    CONSTRAINT check_iap_16 CHECK (array_ndims(iap) = 1 AND array_length(iap, 1) = 16 AND array_position(iap, NULL) IS NULL),
    CONSTRAINT check_iep_7 CHECK (array_ndims(iep) = 1 AND array_length(iep, 1) = 7 AND array_position(iep, NULL) IS NULL)
);
```

#### 3. Bảng quản lý tham số chuẩn hóa (`feature_normalization`)
Lưu trữ véc-tơ kỳ vọng $\boldsymbol{\mu} \in \mathbb{R}^{163}$ và độ lệch chuẩn $\boldsymbol{\sigma} \in \mathbb{R}^{163}$ được tính toán trên toàn bộ 1.000 ảnh chuẩn của kho dữ liệu:

```sql
CREATE TABLE IF NOT EXISTS flower_ir.feature_normalization (
    normalization_id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    method TEXT NOT NULL,
    feature_means DOUBLE PRECISION[] NOT NULL,
    feature_stds DOUBLE PRECISION[] NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    CONSTRAINT check_means_163 CHECK (array_ndims(feature_means) = 1 AND array_length(feature_means, 1) = 163 AND array_position(feature_means, NULL) IS NULL),
    CONSTRAINT check_stds_163 CHECK (array_ndims(feature_stds) = 1 AND array_length(feature_stds, 1) = 163 AND array_position(feature_stds, NULL) IS NULL)
);
```

#### 4. Bảng lưu trữ véc-tơ chuẩn hóa Tầng 2 (`normalized_features`)
Lưu trữ véc-tơ đặc trưng 163 chiều hoàn chỉnh đã qua tiền xử lý chuẩn hóa, sẵn sàng cho các thuật toán so khớp khoảng cách:

```sql
CREATE TABLE IF NOT EXISTS flower_ir.normalized_features (
    image_id BIGINT NOT NULL REFERENCES flower_ir.flower_images(image_id) ON DELETE CASCADE,
    normalization_id BIGINT NOT NULL REFERENCES flower_ir.feature_normalization(normalization_id),
    normalized_vector DOUBLE PRECISION[] NOT NULL,
    PRIMARY KEY (image_id, normalization_id),
    CONSTRAINT check_norm_vector_163 CHECK (array_ndims(normalized_vector) = 1 AND array_length(normalized_vector, 1) = 163 AND array_position(normalized_vector, NULL) IS NULL)
);
```

#### 5. Bảng quản lý phiên bản chỉ mục (`cbir_index_versions` & `cbir_index_entries`)
Bảo đảm tính toàn vẹn hợp đồng phiên bản (Contract Snapshot):

```sql
CREATE TABLE IF NOT EXISTS flower_ir.cbir_index_versions (
    version_id TEXT PRIMARY KEY,
    normalization_id BIGINT NOT NULL REFERENCES flower_ir.feature_normalization(normalization_id),
    contract JSONB NOT NULL,
    manifest JSONB NOT NULL,
    normalizer JSONB NOT NULL,
    active BOOLEAN NOT NULL DEFAULT FALSE,
    published_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flower_ir.cbir_index_entries (
    version_id TEXT NOT NULL REFERENCES flower_ir.cbir_index_versions(version_id) ON DELETE CASCADE,
    image_id BIGINT NOT NULL REFERENCES flower_ir.flower_images(image_id) ON DELETE CASCADE,
    raw_vector DOUBLE PRECISION[] NOT NULL,
    normalized_vector DOUBLE PRECISION[] NOT NULL,
    metadata JSONB NOT NULL,
    PRIMARY KEY (version_id, image_id)
);
```

---

## 3.3. PHƯƠNG PHÁP CHUẨN HÓA ĐẶC TRƯNG CHỌN LỌC (SELECTIVE NORMALIZATION)

### 3.3.1. Định nghĩa (Definition)

**Chuẩn hóa đặc trưng (Feature Normalization)** trong hệ thống CBIR là quá trình biến đổi các giá trị số học của các chiều thuộc tính trong véc-tơ đặc trưng đa chiều về một miền giá trị chung hoặc một phân phối chuẩn có cùng tính chất thống kê, nhằm triệt tiêu sự bất bình đẳng về đơn vị đo lường vật lý và thang độ lớn (scale/magnitude) giữa các thuộc tính thị giác độc lập.

Trong bài toán này, véc-tơ đặc trưng là sự kết hợp không đồng nhất của:
- Biểu đồ tần suất xác suất (Hue-Saturation histogram, LBP histogram).
- Giá trị thống kê không gian màu liên tục (Lab mean, Lab standard deviation).
- Các chỉ số tỷ lệ hình học vô thứ nguyên (Circularity, Aspect Ratio, Solidity, Extent).
- Năng lượng gradient và mật độ cạnh biên (Fourier amplitudes, IAP, IEP).

Do đó, hệ thống áp dụng kỹ thuật **Chuẩn hóa Chọn lọc (Selective Normalization)** thay vì áp dụng một phép biến đổi đồng nhất lên toàn bộ véc-tơ.

---

### 3.3.2. Mục đích (Objectives)

Việc thực hiện chuẩn hóa chọn lọc hướng tới bốn mục tiêu cốt lõi:

1. **Triệt tiêu hiện tượng chi phối không gian (Dominance Effect Elimination):**  
   Các thuộc tính có biên độ đo lường lớn (chẳng hạn thành phần phổ Fourier có biên độ hàng trăm, hoặc độ lệch chuẩn màu Lab dao động từ $0 \to 50$) nếu không được chuẩn hóa sẽ hoàn toàn áp đảo các thuộc tính có biên độ nhỏ (chẳng hạn các tỷ số hình học chỉ dao động trong $[0, 1]$) trong quá trình tính toán khoảng cách khoảng cách số học.

2. **Bảo toàn bản chất hình học của Phân phối Xác suất (Probability Distribution Preservation):**  
   Đối với các thuộc tính dạng biểu đồ tần suất (Histogram màu sắc $\mathbf{F}_1$ và biểu đồ vi kết cấu LBP $\mathbf{F}_3$), các giá trị đã được chuẩn hóa mức cơ sở ($L_1$-norm) để đại diện cho phân phối xác suất rời rạc:
   $$\sum_{k} p_k = 1, \quad p_k \ge 0$$
   Nếu áp dụng chuẩn hóa Z-score tiêu chuẩn lên các chiều này, véc-tơ sẽ xuất hiện các giá trị âm và tổng các phần tử không còn bằng $1$. Điều này phá vỡ tiên đề xác suất và triệt tiêu tính hợp lệ của hàm đo khoảng cách Hellinger. Do đó, mục đích của chuẩn hóa chọn lọc là **giữ nguyên vẹn 106 chiều xác suất của $\mathbf{F}_1$ và $\mathbf{F}_3$**.

3. **Chuẩn hóa phương sai cho các biến liên tục thực (Variance Equalization for Continuous Variables):**  
   Đối với 57 chiều thuộc nhóm giá trị màu Lab ($\mathbf{F}_2$) và nhóm hình dạng/cấu trúc biên ($\mathbf{F}_4$), các chiều này có phân phối giá trị thực không đồng nhất. Mục đích của việc chuẩn hóa Z-score là đưa các chiều này về cùng một thang đo có giá trị kỳ vọng trung bình bằng $0$ ($\mu = 0$) và phương sai bằng $1$ ($\sigma^2 = 1$).

4. **Thiết lập mỏ neo tham số bất biến (Immutable Baseline Anchor):**  
   Đảm bảo toàn bộ tham số thống kê ($\mu, \sigma$) được tính toán và cố định dựa trên tập chuẩn hóa 1.000 ảnh của cơ sở dữ liệu. Mọi ảnh truy vấn mới từ bên ngoài đều phải sử dụng chính xác bộ tham số này để chiếu vào cùng một không gian véc-tơ, ngăn ngừa triệt để hiện tượng rò rỉ dữ liệu hoặc sai lệch chuẩn hóa.

---

### 3.3.3. Công thức Toán học (Mathematical Formulation)

Véc-tơ đặc trưng thô $\mathbf{x} \in \mathbb{R}^{163}$ của một ảnh $i$ được phân rã thành:

$$\mathbf{x}_i = [\mathbf{f}_{1, i} \ (78D) \mid \mathbf{f}_{2, i} \ (18D) \mid \mathbf{f}_{3, i} \ (28D) \mid \mathbf{f}_{4, i} \ (39D)]$$

Tập hợp chỉ số các chiều được phân loại thành hai tập rời nhau:
- **Tập phân phối xác suất:** $\mathcal{I}_{\text{prob}} = [0..77] \cup [96..123]$ (gồm 78 chiều $\mathbf{F}_1$ và 28 chiều $\mathbf{F}_3$, tổng cộng 106 chiều).
- **Tập giá trị thực liên tục:** $\mathcal{I}_{\text{real}} = [78..95] \cup [124..162]$ (gồm 18 chiều $\mathbf{F}_2$ và 39 chiều $\mathbf{F}_4$, tổng cộng 57 chiều).

Thuật toán chuẩn hóa chọn lọc ánh xạ véc-tơ thô $x_{ij}$ thành véc-tơ chuẩn hóa $z_{ij}$ theo quy tắc:

$$z_{ij} = \begin{cases} 
x_{ij}, & \text{nếu } j \in \mathcal{I}_{\text{prob}} \\
\dfrac{x_{ij} - \mu_j}{\sigma_j + \epsilon}, & \text{nếu } j \in \mathcal{I}_{\text{real}}
\end{cases}$$

Trong đó, kỳ vọng mẫu $\mu_j$ và độ lệch chuẩn mẫu $\sigma_j$ của chiều thuộc tính thứ $j$ được tính toán trên toàn bộ tập ảnh cơ sở $N$:

$$\mu_j = \frac{1}{N} \sum_{i=1}^{N} x_{ij}$$

$$\sigma_j = \sqrt{\frac{1}{N} \sum_{i=1}^{N} (x_{ij} - \mu_j)^2}$$

---

### 3.3.4. Giải thích Chi tiết các Tham số

| Ký hiệu | Kiểu dữ liệu | Giá trị / Miền xác định | Giải thích ý nghĩa kỹ thuật |
|:---|:---:|:---:|:---|
| $N$ | Số nguyên | $1.000$ | Tổng số lượng mẫu ảnh hoa trong kho dữ liệu cơ sở được dùng để ước lượng tham số thống kê tổng thể. |
| $i$ | Chỉ số | $1 \le i \le N$ | Chỉ số đại diện cho một ảnh cụ thể trong tập dữ liệu $1.000$ ảnh. |
| $j$ | Chỉ số | $0 \le j \le 162$ | Chỉ số đại diện cho một chiều thuộc tính cụ thể trong véc-tơ đặc trưng 163 chiều. |
| $x_{ij}$ | Số thực (`float64`) | $\mathbb{R}$ | Giá trị đặc trưng thô tại chiều thứ $j$ của ảnh thứ $i$ thu được từ giải thuật trích xuất Tầng 1. |
| $\mu_j$ | Số thực (`float64`) | $\mathbb{R}$ | Giá trị trung bình mẫu (Sample Mean) của chiều đặc trưng thứ $j$ trên toàn bộ 1.000 ảnh. Với các chiều $j \in \mathcal{I}_{\text{prob}}$, hệ thống gán tham số mặc định $\mu_j = 0.0$. |
| $\sigma_j$ | Số thực (`float64`) | $\sigma_j > 0$ | Độ lệch chuẩn tổng thể (Population Standard Deviation với bậc tự do `ddof = 0`) của chiều đặc trưng thứ $j$. Với các chiều $j \in \mathcal{I}_{\text{prob}}$, hệ thống gán tham số mặc định $\sigma_j = 1.0$. |
| $\epsilon$ | Số thực nhỏ | $10^{-12}$ | Hằng số làm mịn số học (Smoothing constant / Machine epsilon), có vai trò ngăn chặn triệt để lỗi chia cho $0$ (Division by Zero) trong trường hợp một thuộc tính có độ phân tán phương sai bằng $0$. |
| $z_{ij}$ | Số thực (`float64`) | $\mathbb{R}$ | Giá trị véc-tơ sau chuẩn hóa Tầng 2. Với 57 chiều thuộc $\mathcal{I}_{\text{real}}$, $z_{ij}$ tuân theo phân phối chuẩn hóa có trung bình bằng $0$ và độ lệch chuẩn bằng $1$ trên toàn kho dữ liệu. |

### Bảng phân bố cấu trúc chuẩn hóa trên 163 chiều:

| Nhóm | Số chiều | Thuộc tính thành phần | Miền giá trị thô | Phương pháp xử lý |
|:---|:---:|:---|:---:|:---:|
| **$\mathbf{F}_1$** | **78** | • H-S Histogram ($12 \times 6 = 72$ bins)<br>• V Histogram (6 bins) | $[0, 1], \sum = 1$ | **Giữ nguyên** ($L_1$-norm) |
| **$\mathbf{F}_2$** | **18** | • Lab Mean 3 vùng bán kính ($3 \times 3 = 9$ chiều)<br>• Lab Std 3 vùng bán kính ($3 \times 3 = 9$ chiều) | $L \in [0, 100]$<br>$a, b \in [-128, 127]$ | **Z-score từng chiều** |
| **$\mathbf{F}_3$** | **28** | • LBP Uniform $(P=8, R=1)$ (10 bins)<br>• LBP Uniform $(P=16, R=2)$ (18 bins) | $[0, 1], \sum = 1$ | **Giữ nguyên** ($L_1$-norm) |
| **$\mathbf{F}_4$** | **39** | • 4 hệ số hình học (Circularity, Solidity, Aspect Ratio, Extent)<br>• 12 thành phần biên độ phổ Fourier ($k=1..12$)<br>• 16 thành phần IAP (4 vành $\times$ 4 góc)<br>• 7 thành phần IEP (5 vành + slope + CV) | $[0, \infty)$ | **Z-score từng chiều** |

---

## 3.4. THUẬT TOÁN ĐO ĐỘ TƯƠNG ĐỒNG VÀ SO KHỚP ĐA ĐẶC TRƯNG

Do 163 chiều đặc trưng được hợp thành từ các cấu trúc dữ liệu khác nhau, việc áp dụng một khoảng cách Euclidean đơn lẻ trên toàn bộ véc-tơ là không chính xác về mặt toán học. Hệ thống áp dụng nguyên tắc **Khoảng cách Phân khối (Block-wise Distance Metric)**:

### 3.4.1. Khoảng cách Hellinger cho Phân phối Xác suất ($\mathbf{F}_1, \mathbf{F}_3$)

Khoảng cách Hellinger giữa hai phân phối xác suất rời rạc $P = (p_1, \dots, p_K)$ và $Q = (q_1, \dots, q_K)$ được định nghĩa:

$$H(P, Q) = \sqrt{\frac{1}{2} \sum_{k=1}^{K} \left(\sqrt{p_k} - \sqrt{q_k}\right)^2}$$

Khoảng cách Hellinger thỏa mãn các tiên đề của một không gian metric, có tính chất đối xứng $H(P, Q) = H(Q, P)$, và bị chặn nghiêm ngặt trong khoảng $[0, 1]$.

- **Khoảng cách Màu sắc Toàn cục ($D_1$):** Kết hợp khoảng cách giữa biểu đồ sắc độ - độ bão hòa (H-S) và biểu đồ độ sáng (V):
  $$D_1 = \frac{1}{2} \left[ H(\mathbf{hs}_q, \mathbf{hs}_t) + H(\mathbf{v}_q, \mathbf{v}_t) \right]$$

- **Khoảng cách Vi kết cấu LBP ($D_3$):** Kết hợp khoảng cách giữa hai biểu đồ LBP đa bán kính:
  $$D_3 = \frac{1}{2} \left[ H(\mathbf{lbp8}_q, \mathbf{lbp8}_t) + H(\mathbf{lbp16}_q, \mathbf{lbp16}_t) \right]$$

---

### 3.4.2. Khoảng cách RMS Biến đổi Phi tuyến cho Giá trị Thực ($\mathbf{F}_2, \mathbf{F}_4$)

Đối với hai véc-tơ giá trị thực đa chiều $\mathbf{a}, \mathbf{b} \in \mathbb{R}^d$, độ phân kỳ bình phương trung bình (Root Mean Square Distance - RMS) phản ánh mức độ lệch trung bình trên mỗi chiều:

$$R(\mathbf{a}, \mathbf{b}) = \sqrt{\frac{1}{d} \sum_{k=1}^{d} (a_k - b_k)^2}$$

Vì $R(\mathbf{a}, \mathbf{b}) \in [0, \infty)$, hàm biến đổi phi tuyến đơn điệu được áp dụng để ánh xạ khoảng cách về không gian bị chặn $[0, 1)$:

$$T(\mathbf{a}, \mathbf{b}) = \frac{R(\mathbf{a}, \mathbf{b})}{1.0 + R(\mathbf{a}, \mathbf{b})}$$

Hàm $T$ bảo toàn tuyệt đối thứ tự khoảng cách ($R_1 < R_2 \iff T_1 < T_2$), đồng thời nén các sai lệch quá lớn (Outlier Damping), giúp khoảng cách không bị bùng nổ khi gặp nhiễu.

- **Khoảng cách Phân bố Không gian Màu Lab ($D_2$):** Tính trên 18 chiều Lab của 3 vùng bán kính tâm - giữa - ngoài:
  $$D_2 = T(\mathbf{f2}_q, \mathbf{f2}_t)$$

- **Khoảng cách Hình dạng và Vi cấu trúc Biên ($D_4$):** Là trung bình cộng khoảng cách biến đổi của 4 khối cấu trúc con:
  $$D_4 = \frac{1}{4} \left[ T(\mathbf{geo}_q, \mathbf{geo}_t) + T(\mathbf{fourier}_q, \mathbf{fourier}_t) + T(\mathbf{iap}_q, \mathbf{iap}_t) + T(\mathbf{iep}_q, \mathbf{iep}_t) \right]$$

---

### 3.4.3. Kết hợp Khoảng cách Tuyến tính có Trọng số Động (Dynamic Weighted Fusion)

Tổng khoảng cách bất tương đồng $D(\mathbf{q}, \mathbf{t})$ giữa ảnh truy vấn $\mathbf{q}$ và ảnh mục tiêu $\mathbf{t}$ là tổ hợp tuyến tính của 4 thành phần khoảng cách độc lập:

$$D(\mathbf{q}, \mathbf{t}) = w_1 D_1 + w_2 D_2 + w_3 D_3 + w_4 D_4$$

Ràng buộc chuẩn hóa trọng số:
$$\sum_{i=1}^{4} w_i = 1, \quad w_i \ge 0 \quad (\forall i \in \{1, 2, 3, 4\})$$

Điểm tương đồng trực quan (Visual Similarity Score $S$) được tính toán phục vụ hiển thị:

$$S(\mathbf{q}, \mathbf{t}) = 1.0 - D(\mathbf{q}, \mathbf{t}) \quad (S \in [0, 1] \text{ hay } [0\%, 100\%])$$

**Chiến lược Xếp hạng và Phá vỡ Hòa điểm (Tie-breaking Rule):**  
Các ứng viên ảnh trong kho CSDL được sắp xếp theo thứ tự ưu tiên:
1. Khoảng cách $D$ tăng dần (khoảng cách nhỏ hơn tương ứng với độ tương đồng cao hơn).
2. Khi khoảng cách $D$ bằng nhau tuyệt đối, hệ thống sử dụng khóa phụ là chỉ số định danh `image_id` tăng dần (`np.lexsort((image_ids, D_all))`). Quy tắc này bảo đảm kết quả trả về luôn có tính tất định (Deterministic) và bất biến qua các lần truy vấn.

---

## 3.5. TỐI ƯU HÓA TÍNH TOÁN VÀ BỘ MÁY TRUY HỒI THỜI GIAN THỰC

### 3.5.1. Phân tích Độ phức tạp Tính toán (Computational Complexity Analysis)

Xét bài toán so khớp với quy mô dữ liệu:
- Số lượng ảnh trong cơ sở dữ liệu: $N = 1.000$ mẫu.
- Số chiều véc-tơ đặc trưng: $D = 163$ chiều.

Khi thực hiện thuật toán Quét véc-tơ hóa Toàn vẹn (Exhaustive Vectorized Scan):
- Độ phức tạp không gian lưu trữ: $\mathcal{O}(N \times D) = 1.000 \times 163 \times 8 \text{ bytes} \approx 1.304 \text{ MB}$. Toàn bộ ma trận đặc trưng chiếm dung lượng rất nhỏ, hoàn toàn có thể lưu trữ thường trực trên bộ nhớ RAM.
- Độ phức tạp thời gian: $\mathcal{O}(N \times D)$. Tổng số phép toán số học dấu phẩy động cần thực hiện cho một truy vấn khoảng $3.2 \times 10^5$ phép tính.

Nhờ tập lệnh song song SIMD (AVX2) tích hợp sẵn trong thư viện NumPy, toàn bộ quá trình tính toán $D_1, D_2, D_3, D_4$ cho toàn bộ 1.000 ảnh diễn ra chỉ trong **$1.15 \text{ ms} - 1.30 \text{ ms}$**.

---

### 3.5.2. Lý giải Lựa chọn Quét Toàn vẹn (Exhaustive Scan) Thay vì Phân cụm (Clustering)

1. **Đạt độ chính xác tuyệt đối (100% Exact Recall):**  
   Các thuật toán phân cụm không gian (như K-Means hay phân vùng cây nhị phân) về bản chất là kỹ thuật tìm kiếm gần đúng (Approximate Nearest Neighbors - ANN). Khi gom dữ liệu vào các cụm, hệ thống chỉ kiểm tra trong các cụm lân cận tâm truy vấn, dẫn tới nguy cơ rất lớn bỏ sót các ảnh hoa nằm ở vùng biên giữa hai cụm (Boundary Effect / False Negatives). Với tập dữ liệu 1.000 ảnh môn học, việc đánh đổi độ chính xác để tiết kiệm dưới $1\text{ ms}$ tính toán là không cần thiết.

2. **Tương thích tuyệt đối với Cơ chế Trọng số Động (Dynamic Weights Support):**  
   Các giải thuật phân cụm hoặc cây chỉ mục đa chiều bắt buộc phải cố định hàm khoảng cách trong không gian $D$ chiều tĩnh từ trước. Trong hệ sinh thái CBIR hiện hành, người dùng được cung cấp quyền năng điều chỉnh thanh trượt trọng số $w_1, w_2, w_3, w_4$ theo thời gian thực (ví dụ: chuyển đổi từ ưu tiên màu sắc sang ưu tiên kết cấu cánh hoa). Mỗi lần thay đổi trọng số, không gian metric bị biến dạng hoàn toàn, làm vô hiệu hóa các cấu trúc cụm được tạo từ trước. Bộ máy quét ma trận trên RAM là giải pháp duy nhất hỗ trợ hoàn hảo trọng số động mà vẫn duy trì tốc độ phản hồi tức thời.

---

## 3.6. THỰC NGHIỆM VÀ ĐÁNH GIÁ ĐỘ ĐÚNG ĐẮN

### 3.6.1. Kiểm thử Tính nhất quán Dữ liệu Từng Bit (Bitwise Round-trip Parity)

Nhằm bảo đảm quá trình ánh xạ dữ liệu giữa tầng ứng dụng và cơ sở dữ liệu quan hệ PostgreSQL không làm thất thoát hoặc sai lệch độ chính xác số học, một kịch bản kiểm thử tự động nghiêm ngặt đã được thực thi:

1. Trích xuất véc-tơ 163 chiều từ 1.000 ảnh gốc và lưu trữ tại `features_163.csv`.
2. Đẩy toàn bộ dữ liệu vào bảng `flower_ir.flower_features` và `flower_ir.normalized_features` trong PostgreSQL thông qua giao thức kết nối nhị phân `psycopg3`.
3. Đọc ngược toàn bộ dữ liệu từ các bảng trong CSDL lên bộ nhớ RAM.
4. Đối chiếu từng phần tử bằng hàm so sánh bitwise `numpy.array_equal()`.

**Kết quả thực nghiệm:**
- Kiểm thử đặc trưng thô Tầng 1: `array_equal(features_csv, features_db) == True` đạt **1.000/1.000 ảnh (100.0%)**.
- Kiểm thử véc-tơ chuẩn hóa Tầng 2: `array_equal(norm_csv, norm_db) == True` đạt **1.000/1.000 ảnh (100.0%)**.
- Kết quả khẳng định cấu trúc mảng `DOUBLE PRECISION[]` của PostgreSQL bảo toàn nguyên vẹn độ chính xác dấu phẩy động 64-bit chuẩn IEEE 754.

---

### 3.6.2. Đo lường Độ trễ Truy vấn Toàn trình (End-to-End Latency Benchmark)

Hệ thống được đo lường thời gian thực thi trên phần cứng chuẩn thực nghiệm (CPU Intel Core i7 12-Cores, 16GB RAM, SSD NVMe). Thời gian trung bình của từng giai đoạn trong quy trình truy vấn một ảnh hoa mới được ghi nhận như sau:

| Giai đoạn thực thi | Tác vụ kỹ thuật | Thời gian trung bình | Tỷ lệ (%) |
|:---|:---|:---:|:---:|
| **Giai đoạn 1** | Tải ảnh truy vấn và chuẩn hóa kích thước $384 \times 384$ px | $4.2\text{ ms}$ | $3.1\%$ |
| **Giai đoạn 2** | Suy luận phân đoạn ngữ nghĩa tự động DeepLabV3+ | $48.5\text{ ms}$ | $35.9\%$ |
| **Giai đoạn 3** | Hậu xử lý mặt nạ nhị phân (khử nhiễu, vá lỗ, liên thông) | $3.1\text{ ms}$ | $2.3\%$ |
| **Giai đoạn 4** | Trích xuất véc-tơ 163 chiều thô ($F_1 \to F_4$) | $72.0\text{ ms}$ | $53.3\%$ |
| **Giai đoạn 5** | Chuẩn hóa Z-score chọn lọc bằng tham số cố định | $0.2\text{ ms}$ | $0.1\%$ |
| **Giai đoạn 6** | **So khớp khoảng cách véc-tơ trên toàn bộ 1.000 ảnh** | **$1.2\text{ ms}$** | **$0.9\%$** |
| **Giai đoạn 7** | Sắp xếp thứ hạng Lexsort và kết xuất kết quả Top-5 | $5.8\text{ ms}$ | $4.3\%$ |
| **Tổng cộng** | **Thời gian đáp ứng toàn trình (End-to-End Query Time)** | **$\approx 135.0\text{ ms}$** | **$100.0\%$** |

Biểu đồ phân rã thời gian cho thấy giai đoạn so khớp dữ liệu trong CSDL chỉ chiếm chưa đầy **$1\%$** tổng thời gian thực thi của hệ thống, khẳng định tầng truy hồi bộ nhớ đạt hiệu năng tối ưu tuyệt đối.

---

### 3.6.3. Kiểm thử Kiểm chứng trên Tập 14 Ảnh Hoa Mới Bất kỳ (Out-of-sample Testing)

Để kiểm chứng tính tổng quát của hệ thống khi tiếp nhận các dữ liệu ngoài tập huấn luyện:
- Tập kiểm thử gồm **14 ảnh hoa chụp thực tế** với các điều kiện góc chụp, ánh sáng và phông nền phức tạp (thư mục `cau_3/14_anh_moi`).
- **Khả năng phân đoạn:** Pipeline DeepLabV3+ kết hợp bộ lọc hậu xử lý hình thái học tách chính xác 14/14 bông hoa chính ra khỏi nền cành lá, sinh ra mặt nạ nhị phân $384 \times 384$ đạt độ hoàn thiện cao.
- **Tính ổn định của trích xuất:** $100\%$ các véc-tơ 163 chiều sinh ra đều có giá trị hữu hạn, không xuất hiện các lỗi giá trị dị thường (`NaN`, `Inf`).
- **Chất lượng truy hồi:** Kết quả Top-5 trả về trên giao diện Web thể hiện rõ độ tương đồng cao về màu sắc cánh hoa (thông qua $D_1$), vi cấu trúc vân cánh (thông qua $D_3$) và hình dạng tổng thể (thông qua $D_4$).
