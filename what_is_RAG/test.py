import argparse
import logging
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings

load_dotenv()
DEFAULT_WORDS = ["apple", "banana", "computer"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Log lower-dimensional OpenAI embeddings for input words."
    )
    parser.add_argument(
        "words",
        nargs="*",
        default=DEFAULT_WORDS,
        help="Words to embed. Uses sample words when omitted.",
    )
    parser.add_argument(
        "--dimensions",
        type=int,
        default=10,
        help="Number of dimensions returned for each embedding (default: 10).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small",
        dimensions=args.dimensions,
    )

    vectors = embeddings.embed_documents(args.words)

    for word, vector in zip(args.words, vectors):
        print("word=%r dimensions=%d embedding=%s", word, len(vector), vector)
        print("\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
