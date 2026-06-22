import csv
import logging
from pathlib import Path
from src.pipeline.run_ingestion import run_pipeline

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

def ingest_from_csv(csv_path: str):
    if not csv_path:
        logging.error("No CSV path provided.")
        return
    
    file_path = Path(csv_path)

    if not file_path.exists():
        logging.error(f"File not found: {csv_path}")
        return

    success_count = 0
    failure_count = 0

    with open(file_path, mode="r", encoding="utf-8-sig") as f:
        reader = csv.DictReader(f)

        for row in reader:
            doc_id = row.get("document_id")
            md_path = row.get("markdown_path")
            url = row.get("frontend_url")

            if not doc_id or not md_path or not url:
                logging.warning(f"Skipping invalid row: {row}")
                continue

            logging.info(f"Ingesting {doc_id}")

            try:
                run_pipeline(
                    mdfile_path=md_path,
                    frontend_url=url,
                    document_id=doc_id,
                )
                success_count += 1
                logging.info(f"Completed {doc_id}")

            except Exception as e:
                failure_count += 1
                logging.error(f"Failed {doc_id}: {e}")

    logging.info(f"Done. Success: {success_count}, Failed: {failure_count}")