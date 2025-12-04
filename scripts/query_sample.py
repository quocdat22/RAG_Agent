import argparse

from rag.pipeline import answer_query


def main():
    parser = argparse.ArgumentParser(description="Query RAG pipeline.")
    parser.add_argument("query", help="User question")
    args = parser.parse_args()

    result = answer_query(args.query)
    print("Answer:")
    print(result.answer)
    print("\nSources:")
    for i, src in enumerate(result.sources, start=1):
        print(f"{i}. {src}")


if __name__ == "__main__":
    main()


