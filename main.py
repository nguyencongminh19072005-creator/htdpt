import sys
import os
import argparse
import subprocess
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    if sys.stdout:
        sys.stdout.reconfigure(encoding='utf-8')
    if sys.stderr:
        sys.stderr.reconfigure(encoding='utf-8')
except Exception:
    pass


def run_app():
    """Launch the Streamlit web application."""
    app_path = PROJECT_ROOT / "flower_cbir" / "app.py"
    cmd = [sys.executable, "-m", "streamlit", "run", str(app_path)]
    print(f"[Main] Khởi chạy giao diện demo: {' '.join(cmd)}")
    raise SystemExit(subprocess.run(cmd).returncode)


def main():
    parser = argparse.ArgumentParser(description="Hệ thống CBIR Tìm kiếm Ảnh Bông hoa 163D (PTIT)")
    parser.add_argument("--index", action="store_true", help="Tạo mask DeepLabv3, trích 163D, xuất CSV và đồng bộ PostgreSQL")
    parser.add_argument("--test", action="store_true", help="Chạy kiểm tra bất biến toán học và ablation study")
    parser.add_argument("--search", type=str, default=None, help="Đường dẫn ảnh truy vấn để tìm kiếm qua CLI")
    parser.add_argument("--app", action="store_true", help="Khởi chạy giao diện web Streamlit")
    parser.add_argument('--evaluate', action='store_true', help='Run reproducible validation/test evaluation and weight selection')
    parser.add_argument('--verify-sync', action='store_true', help='Verify bitwise parity of DeepLabV3, query, CSV and PostgreSQL')
    parser.add_argument('--flower-box',nargs=4,type=float,metavar=('X','Y','WIDTH','HEIGHT'),help='Flower bounding box; expanded by 1.10 and padded per report')

    args = parser.parse_args()
    if args.verify_sync:
        from flower_cbir.services.verification import verify_synchronization
        verify_synchronization()
        return
    if args.evaluate:
        from flower_cbir.evaluation.experiment import run_experiment
        run_experiment()
        return

    if args.index:
        from flower_cbir.services.index_dataset import build_or_load_index
        from flower_cbir.services.database import FlowerDatabase
        from flower_cbir.services.csv_tools import export_csv
        print("[Main] Đang xây dựng bộ chỉ mục dữ liệu...")
        data=build_or_load_index()
        export_csv(data,PROJECT_ROOT/'trich_rut_minh/features_163.csv',PROJECT_ROOT/'trich_rut_minh/extraction_manifest_163.json')
        database=FlowerDatabase()
        try:
            print(database.publish_index(data))
        finally:database.close()
        print("[Main] Đã đồng bộ đặc trưng, normalizer và vector PostgreSQL; đã xuất CSV cùng phiên bản.")
        return

    if args.test:
        from flower_cbir.services.search import FlowerSearchEngine
        from flower_cbir.evaluation.evaluate import run_correctness_checks, run_ablation_study
        print("[Main] Đang khởi tạo Search Engine...")
        engine = FlowerSearchEngine()
        print("\n--- 1. KIỂM TRA BẤT BIẾN TOÁN HỌC ---")
        report = run_correctness_checks(engine)
        for t in report["tests"]:
            mark = "✅ PASS" if t["passed"] else "❌ FAIL"
            print(f"{mark} | {t['name']}: {t['detail']}")

        print("\n--- 2. ABLATION STUDY (ẢNH MẪU 0) ---")
        abl = run_ablation_study(engine, query_idx=0)
        for name, data in abl.items():
            top1 = data["top5"][0]["filename"]
            sim1 = data["top5"][0]["similarity"]
            print(f"- {name:<30}: Top-1={top1:<18} Sim={sim1*100:.1f}% ({data['search_ms']:.2f}ms)")
        if not all(item['passed'] for item in report['tests']):
            raise SystemExit(1)
        return

    if args.search:
        from flower_cbir.services.search import FlowerSearchEngine
        img_path = Path(args.search)
        if not img_path.exists():
            print(f"[Error] Không tìm thấy file: {img_path}")
            return
        engine = FlowerSearchEngine()
        print(f"[Main] Đang truy vấn ảnh: {img_path.name}...")
        res = engine.search(img_path,flower_box=args.flower_box)
        print(f"\n--- KẾT QUẢ TOP 5 (Thời gian xử lý: {res['timings']['total_backend_ms']:.1f}ms) ---")
        for item in res["results"]:
            print(f"Rank #{item['rank']}: {item['filename']} | Similarity: {item['similarity']*100:.1f}% | Distance: {item['distance']:.4f}")
            print(f"        (D1={item['D1']:.4f}, D2={item['D2']:.4f}, D3={item['D3']:.4f}, D4={item['D4']:.4f})")
        return

    # Default action: launch app
    run_app()


if __name__ == "__main__":
    main()
