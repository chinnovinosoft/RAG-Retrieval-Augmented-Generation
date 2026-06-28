import os
import uuid
from dotenv import load_dotenv

from langsmith import Client, traceable, get_current_run_tree

from rag_graph import graph

load_dotenv()

ls_client = Client()
QUESTION = "give details about planning and operation of information security management system as per ISO 27001 standard"


@traceable(
    name="RAG Pipeline",
    tags=["v1"],
    metadata={"pipeline": "simple-rag"}
)
def run_rag(question: str) -> dict:
    result = graph.invoke({"question": question})
    run_id = str(get_current_run_tree().id)
    return {"answer": result["answer"], "run_id": run_id}


def add_feedback(question: str):
    output = run_rag(question)
    run_id = output["run_id"]

    ls_client.create_feedback(
        run_id=run_id,
        key="user_rating",
        score=1.0,
        comment="Answer was accurate and concise"
    )

    ls_client.create_feedback(
        run_id=run_id,
        key="relevance",
        score=0.9,
        comment="Retrieved context was highly relevant"
    )

    print(f"Feedback submitted for run: {run_id}")


if __name__ == "__main__":

    print("\n--- @traceable ---")
    output = run_rag(QUESTION)
    print(f"run_id: {output['run_id']}")
    print(output["answer"])

    print("\n--- Feedback ---")
    add_feedback(QUESTION)

