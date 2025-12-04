"""Test script to check LLM response structure."""
import sys
from pathlib import Path

# Add project root to path
ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from rag.llm import generate_answer

if __name__ == "__main__":
    # Simple test
    test_query = "Xin chào, bạn là ai?"
    test_contexts = ["Đây là một tài liệu test về RAG system."]
    
    print("Testing LLM...")
    print(f"Query: {test_query}")
    print(f"Contexts: {test_contexts}")
    print("-" * 50)
    
    try:
        answer = generate_answer(test_query, test_contexts)
        print(f"Answer: {answer}")
        print(f"Answer type: {type(answer)}")
        print(f"Answer length: {len(answer) if answer else 0}")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

