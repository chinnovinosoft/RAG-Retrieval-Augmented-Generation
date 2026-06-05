import json
import os
import pandas as pd

from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_openai import ChatOpenAI

from deepeval.test_case import LLMTestCase
from deepeval.metrics import (
    AnswerRelevancyMetric,
    FaithfulnessMetric,
    ContextualPrecisionMetric,
    ContextualRecallMetric,
    ContextualRelevancyMetric
)

load_dotenv()

from pinecone import Pinecone
from langchain_openai import OpenAIEmbeddings
from langchain_pinecone import PineconeVectorStore

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

index_name = "labour-index"

vectorstore = PineconeVectorStore(
    index_name=index_name,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

with open("dataset.json", "r", encoding="utf-8") as f:
    evaluation_dataset = json.load(f)

print(f"Loaded {len(evaluation_dataset)} evaluation samples")

answer_relevancy = AnswerRelevancyMetric(
    threshold=0.7,
    model="gpt-4o"
)

faithfulness = FaithfulnessMetric(
    threshold=0.7,
    model="gpt-4o"
)

context_precision = ContextualPrecisionMetric(
    threshold=0.7,
    model="gpt-4o"
)

context_recall = ContextualRecallMetric(
    threshold=0.7,
    model="gpt-4o"
)

context_relevancy = ContextualRelevancyMetric(
    threshold=0.7,
    model="gpt-4o"
)

results = []

for idx, item in enumerate(evaluation_dataset, start=1):

    question = item["question"]
    expected_answer = item["expected_answer"]

    print(f"\n[{idx}/{len(evaluation_dataset)}] Evaluating...")
    print(f"Question: {question}")

    retrieved_docs = retriever.invoke(question)

    retrieval_context = [
        doc.page_content
        for doc in retrieved_docs
    ]

    context_text = "\n\n".join(retrieval_context)

    prompt = f"""
        Answer the question using only the provided context.

        Context:
        {context_text}

        Question:
        {question}
    """

    response = llm.invoke(prompt)

    actual_answer = response.content

    test_case = LLMTestCase(
        input=question,
        actual_output=actual_answer,
        expected_output=expected_answer,
        retrieval_context=retrieval_context
    )

    answer_relevancy.measure(test_case)
    faithfulness.measure(test_case)
    context_precision.measure(test_case)
    context_recall.measure(test_case)
    context_relevancy.measure(test_case)

    results.append({
        "Question": question,
        "Expected Answer": expected_answer,
        "Generated Answer": actual_answer,
        "Answer Relevancy": answer_relevancy.score,
        "Faithfulness": faithfulness.score,
        "Context Precision": context_precision.score,
        "Context Recall": context_recall.score,
        "Context Relevancy": context_relevancy.score,
        "Answer Relevancy Reason": answer_relevancy.reason,
        "Faithfulness Reason": faithfulness.reason,
        "Context Precision Reason": context_precision.reason,
        "Context Recall Reason": context_recall.reason,
        "Context Relevancy Reason": context_relevancy.reason
    })

df = pd.DataFrame(results)

df.to_excel(
    "rag_evaluation_results.xlsx",
    index=False
)

summary = {
    "Average Answer Relevancy": df["Answer Relevancy"].mean(),
    "Average Faithfulness": df["Faithfulness"].mean(),
    "Average Context Precision": df["Context Precision"].mean(),
    "Average Context Recall": df["Context Recall"].mean(),
    "Average Context Relevancy": df["Context Relevancy"].mean(),
}

print("\n" + "=" * 60)
print("EVALUATION SUMMARY")
print("=" * 60)

for metric, value in summary.items():
    print(f"{metric}: {value:.4f}")

print("\nSaved results to rag_evaluation_results.xlsx")