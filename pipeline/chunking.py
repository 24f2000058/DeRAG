from langchain_text_splitters import RecursiveCharacterTextSplitter

_SPLITTER = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)


def chunk_text(normalized_text: str) -> list[str]:
    return _SPLITTER.split_text(normalized_text)
