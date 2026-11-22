CREATE SCHEMA IF NOT EXISTS flower_ir;

CREATE TABLE IF NOT EXISTS flower_ir.feature_config (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    parameters JSONB NOT NULL,
    created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flower_ir.images (
    id BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    filename TEXT NOT NULL UNIQUE,
    image_path TEXT NOT NULL,
    mask_path TEXT NOT NULL,
    image_sha256 CHAR(64) NOT NULL,
    mask_sha256 CHAR(64) NOT NULL,
    width INTEGER NOT NULL CHECK (width = 384),
    height INTEGER NOT NULL CHECK (height = 384),
    flower_pixels INTEGER NOT NULL CHECK (flower_pixels > 0 AND flower_pixels < 384 * 384),
    updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS flower_ir.color_features (
    image_id BIGINT NOT NULL REFERENCES flower_ir.images(id) ON DELETE CASCADE,
    config_id BIGINT NOT NULL REFERENCES flower_ir.feature_config(id),
    hs_hist DOUBLE PRECISION[] NOT NULL,
    value_hist DOUBLE PRECISION[] NOT NULL,
    lab_mean DOUBLE PRECISION[] NOT NULL,
    lab_std DOUBLE PRECISION[] NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id, config_id),
    CONSTRAINT hs_hist_48 CHECK (array_ndims(hs_hist) = 1 AND array_length(hs_hist, 1) = 48 AND array_position(hs_hist, NULL) IS NULL),
    CONSTRAINT value_hist_16 CHECK (array_ndims(value_hist) = 1 AND array_length(value_hist, 1) = 16 AND array_position(value_hist, NULL) IS NULL),
    CONSTRAINT lab_mean_9 CHECK (array_ndims(lab_mean) = 1 AND array_length(lab_mean, 1) = 9 AND array_position(lab_mean, NULL) IS NULL),
    CONSTRAINT lab_std_9 CHECK (array_ndims(lab_std) = 1 AND array_length(lab_std, 1) = 9 AND array_position(lab_std, NULL) IS NULL)
);

CREATE INDEX IF NOT EXISTS images_image_sha256_idx ON flower_ir.images(image_sha256);
CREATE INDEX IF NOT EXISTS color_features_config_idx ON flower_ir.color_features(config_id);

COMMENT ON TABLE flower_ir.images IS 'One source image and its aligned flower mask; no species labels are used.';
COMMENT ON TABLE flower_ir.color_features IS 'Four color descriptors from csdldpt (2).pdf; 82 numeric dimensions per image.';
COMMENT ON COLUMN flower_ir.color_features.hs_hist IS 'H-major flattened 12 hue x 4 saturation histogram; sum=1.';
COMMENT ON COLUMN flower_ir.color_features.value_hist IS '16-bin HSV Value histogram; sum=1.';
COMMENT ON COLUMN flower_ir.color_features.lab_mean IS 'Lab means for center, middle, outer radial flower regions; L,a,b per region.';
COMMENT ON COLUMN flower_ir.color_features.lab_std IS 'Population Lab standard deviations for the same 3 radial regions.';

-- Current feature set from csdldpt (4).pdf, Table 2.1 (PDF page 35).
-- color_features above is retained only to preserve the earlier 82-D trial.
CREATE TABLE IF NOT EXISTS flower_ir.full_features_163 (
    image_id BIGINT NOT NULL REFERENCES flower_ir.images(id) ON DELETE CASCADE,
    config_id BIGINT NOT NULL REFERENCES flower_ir.feature_config(id),
    f1_global_color DOUBLE PRECISION[] NOT NULL,
    f2_spatial_lab DOUBLE PRECISION[] NOT NULL,
    f3_multiradius_lbp DOUBLE PRECISION[] NOT NULL,
    f4_shape_structure DOUBLE PRECISION[] NOT NULL,
    extracted_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (image_id, config_id),
    CONSTRAINT full_f1_78 CHECK (array_ndims(f1_global_color) = 1 AND array_length(f1_global_color, 1) = 78 AND array_position(f1_global_color, NULL) IS NULL),
    CONSTRAINT full_f2_18 CHECK (array_ndims(f2_spatial_lab) = 1 AND array_length(f2_spatial_lab, 1) = 18 AND array_position(f2_spatial_lab, NULL) IS NULL),
    CONSTRAINT full_f3_28 CHECK (array_ndims(f3_multiradius_lbp) = 1 AND array_length(f3_multiradius_lbp, 1) = 28 AND array_position(f3_multiradius_lbp, NULL) IS NULL),
    CONSTRAINT full_f4_39 CHECK (array_ndims(f4_shape_structure) = 1 AND array_length(f4_shape_structure, 1) = 39 AND array_position(f4_shape_structure, NULL) IS NULL)
);

CREATE INDEX IF NOT EXISTS full_features_163_config_idx ON flower_ir.full_features_163(config_id);
COMMENT ON TABLE flower_ir.full_features_163 IS 'Current 163-D F1|F2|F3|F4 descriptor from csdldpt (4).pdf, Table 2.1.';
