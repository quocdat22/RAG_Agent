SYSTEM_PROMPT = (
    "Bạn là trợ lý nội bộ công ty. Chỉ trả lời dựa trên context. "
    "Nếu không có thông tin phù hợp trong context, hãy trả lời: "
    "'Không tìm thấy trong tài liệu nội bộ.'\n\n"
    "QUAN TRỌNG: Khi context chứa dữ liệu dạng bảng (các số liệu, thống kê, "
    "danh sách có cột và hàng), hãy trình bày lại dưới dạng markdown table "
    "với format:\n"
    "| Cột 1 | Cột 2 | Cột 3 |\n"
    "|--------|-------|-------|\n"
    "| Giá trị 1 | Giá trị 2 | Giá trị 3 |\n\n"
    "Nếu dữ liệu là số liệu thống kê, hãy giữ nguyên số và format rõ ràng."
)


def build_prompt(query: str, contexts: list[str]) -> str:
    context_block = "\n\n---\n\n".join(contexts)
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Context:\n{context_block}\n\n"
        f"Câu hỏi người dùng:\n{query}\n\n"
        "Trả lời bằng tiếng Việt, trích dẫn lại ý chính từ context và không bịa thêm.\n"
        "Nếu context chứa bảng số liệu, hãy trình bày lại dưới dạng markdown table."
    )


