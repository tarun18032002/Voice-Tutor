from dotenv import load_dotenv
from livekit.agents import JobContext, WorkerOptions, cli


load_dotenv(".env.local")



if __name__ == "__main__":
    from src.query.tutor_agent import entrypoint
    cli.run_app(
        WorkerOptions(entrypoint_fnc=entrypoint)
    )