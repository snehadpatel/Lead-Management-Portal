"""
Lume AI — Master Orchestration Pipeline
Automates: Spark ETL -> ML Training -> BI Export -> System Manifest
"""

import os
import sys
import time
from pathlib import Path
from datetime import datetime

# Ensure src is in path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

# Set correct JAVA_HOME for Local Spark compatibility
os.environ["JAVA_HOME"] = "/opt/homebrew/opt/openjdk@11/libexec/openjdk.jdk/Contents/Home"

from lume_platform.spark.pipeline import run_default_etl
from lume_platform.ml.training import train_all
from scripts.bi_export import BIExporter
from lume_platform.config import ensure_dirs, MODELS_DIR

def run_full_system(sample: float | None = 0.2):
    print("="*60)
    print(f"🚀 LUME AI MASTER PIPELINE — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("="*60)
    
    ensure_dirs()
    start_time = time.time()
    
    # 1. Spark ETL (The Heavy Lifting)
    print(f"\n[STEP 1/3] Running Big Data ETL (Apache Spark, Sample={sample})...")
    try:
        run_default_etl(sample=sample)
        print("✅ Spark ETL Complete.")
    except Exception as e:
        print(f"❌ Spark ETL Failed: {e}")
        # We continue as training has CSV fallbacks
    
    # 2. ML Training (The Intelligence)
    print("\n[STEP 2/3] Running ML Model Training (Spark-Linked)...")
    try:
        paths = train_all()
        print(f"✅ ML Training Complete. Models saved to: {MODELS_DIR}")
        for k, v in paths.items():
            print(f"   - {k}: {Path(v).name}")
    except Exception as e:
        print(f"❌ ML Training Failed: {e}")
        return

    # 3. BI Export (The Insight)
    print("\n[STEP 3/3] Exporting Data for BI Tools (Tableau/Power BI)...")
    try:
        exporter = BIExporter()
        exporter.export_all()
        print("✅ BI Export Complete.")
    except Exception as e:
        print(f"❌ BI Export Failed: {e}")

    duration = round(time.time() - start_time, 2)
    print("\n" + "="*60)
    print(f"🎉 FULL SYSTEM REFRESH COMPLETE IN {duration}s")
    print("="*60)
    print("\nNext Steps:")
    print("1. Launch API:  uvicorn api.main_enhanced:app --reload")
    print("2. Launch Dashboard: streamlit run streamlit_app_fintech.py")

if __name__ == "__main__":
    # In a real startup, we could pass args for 'full' vs 'sample'
    run_full_system()
