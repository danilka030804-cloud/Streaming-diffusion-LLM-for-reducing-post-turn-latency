def chunk_text_by_words(text: str, chunk_size: int = 5) -> list[str]:
    words = text.replace('\n', ' ').split()
    
    chunks = []
    for i in range(0, len(words), chunk_size):
        
        chunk_words = words[i:i + chunk_size]
        
        chunk_text = " ".join(chunk_words)
        
        chunks.append(chunk_text)
        
    return chunks