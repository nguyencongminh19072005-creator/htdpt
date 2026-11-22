import time
import hashlib
import numpy as np
from pathlib import Path

from flower_cbir.core.preprocess import load_image
from flower_cbir.core.distance import compute_overall_distance
from flower_cbir.settings import default_weights, validate_weights, load_config
from flower_cbir.core.pipeline import extract_image
from flower_cbir.core.contract import verify_contract, file_sha256
from flower_cbir.core.segmentation import clean_mask


class FlowerSearchEngine:
    """
    CBIR Search Engine for 163-dimensional flower image retrieval.
    Performs full vector matching against dataset index.
    """
    def __init__(self, index_data: dict = None):
        if index_data is None:
            from flower_cbir.services.database import FlowerDatabase
            database = FlowerDatabase()
            try:
                index_data = database.load_index()
            finally:
                database.close()

        self.metadata = index_data["metadata"]
        self.raw_matrix = index_data["raw_features"]
        self.normalized_matrix = index_data["normalized_features"]
        self.normalizer = index_data["normalizer"]
        self.num_images = len(self.metadata)
        self.version_id = index_data.get('version_id', 'unversioned')
        self.contract = index_data.get('contract')
        self.manifest = index_data.get('manifest',{})
        if self.contract is not None:
            verify_contract(self.contract)
        if self.raw_matrix.shape != (self.num_images, 163) or self.normalized_matrix.shape != self.raw_matrix.shape or self.num_images == 0:
            raise ValueError('Index shape does not match metadata.')
        if not np.isfinite(self.raw_matrix).all() or not np.isfinite(self.normalized_matrix).all():
            raise ValueError('Index contains nonfinite values.')
        self.image_ids = np.array([item['image_id'] for item in self.metadata])
        self.pixel_lookup = {item['pixel_sha256']: item for item in self.metadata if 'pixel_sha256' in item}

    def match_source(self, source):
        image = load_image(source)
        if image.shape != (384,384,3):
            return None
        return self.pixel_lookup.get(hashlib.sha256(image.tobytes()).hexdigest())

    def search_by_vector(self, query_norm_vector: np.ndarray,
                         weights=None,
                         top_k: int = 5,
                         exclude_filename: str = None):
        """
        Search directly given a normalized 163-d query vector.
        Vectorized across all dataset images via NumPy.
        """
        t0 = time.perf_counter()
        if isinstance(top_k, bool) or not isinstance(top_k, (int, np.integer)) or not 1 <= top_k <= self.num_images:
            raise ValueError('top_k must be an integer between 1 and the gallery size.')
        weights = default_weights() if weights is None else validate_weights(weights)
        D_all, S_all, details = compute_overall_distance(
            query_norm_vector, self.normalized_matrix, weights=weights
        )

        # Sort order: D ascending. Tie-break: image_id ascending
        image_ids = self.image_ids
        # lexsort sorts by last key primary, first key secondary -> (image_ids, D_all)
        sorted_indices = np.lexsort((image_ids, D_all))

        results = []
        rank = 1
        for idx in sorted_indices:
            meta = self.metadata[idx]
            if exclude_filename and meta["filename"] == exclude_filename:
                continue

            w1, w2, w3, w4 = weights
            d1_val = float(details["D1"][idx])
            d2_val = float(details["D2"][idx])
            d3_val = float(details["D3"][idx])
            d4_val = float(details["D4"][idx])

            item = {
                "rank": rank,
                "image_id": meta["image_id"],
                "filename": meta["filename"],
                "image_path": meta["image_path"],
                "mask_path": meta["mask_path"],
                "distance": float(D_all[idx]),
                "similarity": float(S_all[idx]),
                "D1": d1_val,
                "D2": d2_val,
                "D3": d3_val,
                "D4": d4_val,
                "contributions": {
                    "w1_D1": float(w1 * d1_val),
                    "w2_D2": float(w2 * d2_val),
                    "w3_D3": float(w3 * d3_val),
                    "w4_D4": float(w4 * d4_val),
                },
                "f4_details": {
                    "scalars": float(details["d4_scalars"][idx]),
                    "fourier": float(details["d4_fourier"][idx]),
                    "iap": float(details["d4_iap"][idx]),
                    "iep": float(details["d4_iep"][idx]),
                }
            }
            results.append(item)
            rank += 1
            if rank > top_k:
                break

        return results, (time.perf_counter() - t0) * 1000.0

    def rerank(self, previous_result, weights=None, top_k=5, leave_one_out=False):
        if self.contract is not None:
            verify_contract(self.contract)
        if previous_result.get('version_id') != self.version_id:
            raise ValueError('Query belongs to another index version. Run a new query.')
        weights = default_weights() if weights is None else validate_weights(weights)
        identity = previous_result.get('matched_filename')
        results, elapsed = self.search_by_vector(previous_result['query_info']['norm_vector'], weights=weights, top_k=top_k, exclude_filename=identity if leave_one_out else None)
        return {**previous_result, 'results': results, 'weights': list(weights), 'timings': {'preprocess_ms': 0.0, 'segmentation_ms': 0.0, 'extraction_ms': 0.0, 'normalization_ms': 0.0, 'search_ms': elapsed, 'total_backend_ms': elapsed}, 'reused_query_features': True}

    def search(self, query_source,
               query_mask: np.ndarray = None,
               grabcut_rect=None,
               crop_box=None,
               weights=None,
               top_k: int = 5,
               leave_one_out: bool = False,
               query_filename: str = None,
               flower_box=None,
               use_gallery_mask: bool = True):
        """
        Full end-to-end query workflow:
        1. Preprocess query image -> 384x384 RGB.
        2. Use the cached model mask for an unchanged gallery image; otherwise
           run the same DeepLabV3+ model and mask postprocessing.
        3. Feature extraction (163 dimensions) + explanation signals.
        4. Normalization.
        5. Vectorized distance search against index.
        """
        if grabcut_rect is not None:
            raise ValueError('GrabCut is outside the report pipeline. Use flower_box for framing.')
        if self.contract is not None:
            verify_contract(self.contract)
            from flower_cbir.core.deeplab import get_segmenter
            if get_segmenter().info != self.manifest['model_runtime']:
                raise ValueError('DeepLabV3+ runtime differs from this index. Synchronize before querying.')
            if flower_box is None and crop_box is None:
                gallery_item = self.match_source(query_source)
                if gallery_item is not None and query_mask is None and use_gallery_mask:
                    from PIL import Image
                    with Image.open(gallery_item['mask_path']) as saved:
                        query_mask = np.asarray(saved.convert('L'))
        weights = default_weights() if weights is None else validate_weights(weights)
        total_start = time.perf_counter()
        query_info = extract_image(query_source,supplied_mask=query_mask,flower_box=flower_box,crop_box=crop_box)
        timings = dict(query_info.pop('timings'))
        img_rgb = query_info['image_rgb']
        matched = self.pixel_lookup.get(hashlib.sha256(img_rgb.tobytes()).hexdigest())
        if self.contract is not None and query_mask is not None:
            if matched is None:
                raise ValueError('A supplied mask is allowed only for a matching gallery image.')
            from PIL import Image
            if file_sha256(matched['mask_path']) != matched['mask_sha256']:
                raise ValueError('The stored gallery mask changed after index publication.')
            with Image.open(matched['mask_path']) as saved:
                expected_mask = clean_mask(np.asarray(saved.convert('L')))
            if not np.array_equal(query_info['mask_binary'],expected_mask):
                raise ValueError('The supplied query mask differs from the stored model mask.')
            query_info['segmentation']={'method':'cached_deeplabv3plus_v2_mask'}
        t0 = time.perf_counter()
        norm_vector = self.normalizer.transform(query_info['raw_vector'])
        timings["normalization_ms"] = (time.perf_counter() - t0) * 1000.0

        # Step 5: Search
        exclude_name = matched['filename'] if leave_one_out and matched else None
        results, t_search_ms = self.search_by_vector(
            norm_vector, weights=weights, top_k=top_k, exclude_filename=exclude_name
        )
        timings["search_ms"] = t_search_ms
        timings["total_backend_ms"] = (time.perf_counter() - total_start) * 1000.0

        query_info['norm_vector'] = norm_vector

        return {
            "query_info": query_info,
            "results": results,
            "timings": timings,
            "version_id": self.version_id,
            "weights": list(weights),
            "matched_filename": matched['filename'] if matched else None,
            "reused_query_features": False,
        }
