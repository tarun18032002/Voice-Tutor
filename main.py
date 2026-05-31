from dotenv import load_dotenv
from livekit.agents import WorkerOptions, cli
import os

# Load environment variables from .env
load_dotenv()


def ingestion_job():
    from src.injestion.injestion_pipeline import IngestionPipeline

    # Load environment variables
    load_dotenv()

    pipeline = IngestionPipeline()

    file_path = "test/chapter-1(10th std).pdf"

    if os.path.exists(file_path):
        print(f"File found: {file_path}")
    else:
        print(f"File not found: {file_path}")
        return

    pipeline.ingest(file_path)


if __name__ == "__main__":
    # Debug: verify env variables are loaded
    print("LIVEKIT_URL =", os.getenv("LIVEKIT_URL"))
    print("LIVEKIT_API_KEY =", os.getenv("LIVEKIT_API_KEY"))

    from src.query.tutor_agent import entrypoint

    cli.run_app(
        WorkerOptions(
            entrypoint_fnc=entrypoint
        )
    )

    # Uncomment to run ingestion instead of agent
    # ingestion_job()