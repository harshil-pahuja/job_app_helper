"""Demo script for testing the agent locally."""
from backend.agent import run_agent_analysis


if __name__ == "__main__":
    # Example usage
    prompt = "Use the README tool and explain what this project does in one short paragraph."
    response = run_agent_analysis(prompt)
