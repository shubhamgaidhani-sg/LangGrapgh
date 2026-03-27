from typing import Any, Dict

from Graph.chains.generation import generation_chain
from Graph.state import GraphState


# new *
def generate(state: GraphState) -> Dict[str, Any]:
    print("---GENERATE---")

    question = state["question"]
    documents = state["documents"]

    generation = generation_chain.invoke({
        "context": documents,
        "question": question
    })

    return {
        "documents": documents,
        "question": question,
        "generation": generation
    }