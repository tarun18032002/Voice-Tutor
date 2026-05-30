from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli

load_dotenv(".env.local")
import os


def ingestion_job():
    from src.injestion.injestion_pipeline import IngestionPipeline
    load_dotenv(".env.local")  # Load environment variables from .env file

    pipeline = IngestionPipeline()

    # Example file path, replace with actual path or pass as argument
    file_path = "test/chapter-1(10th std).pdf"
    if os.path.exists(file_path):
        print(f"File found: {file_path}")
    else:
        print("File not found")
        return
    pipeline.ingest(file_path)



if __name__ == "__main__":
    from src.query.tutor_agent import entrypoint
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )
    # ingestion_job()