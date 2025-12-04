import argparse

from ingestion.pipeline import run_ingestion


def main():
    parser = argparse.ArgumentParser(description="Ingest a folder of documents into Chroma.")
    parser.add_argument("folder", help="Path to folder containing documents")
    args = parser.parse_args()

    count = run_ingestion(args.folder)
    print(f"Ingested {count} chunks from {args.folder}")


if __name__ == "__main__":
    main()


