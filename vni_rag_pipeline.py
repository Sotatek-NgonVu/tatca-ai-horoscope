# =============================================================================
# VNI-Windows PDF → Clean UTF-8 Unicode → LangChain Chunks
# RAG Preprocessing Pipeline for Legacy Vietnamese Documents
# =============================================================================
# Install dependencies:
#   pip install pdfplumber langchain langchain-text-splitters
# =============================================================================

import re
import pdfplumber
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter


# =============================================================================
# STEP 1: VNI-Windows → Unicode Mapping
# =============================================================================
# VNI encodes Vietnamese using two bytes: a base Latin letter + a high modifier byte.
# When read as Latin-1, the modifier byte appears as a garbled Latin character.
# This table maps those garbled sequences to the correct Unicode character.
#
# Format: "garbled_string" -> "correct_unicode_char"
# =============================================================================

VNI_TO_UNICODE_MAP = {
    # ── Uppercase ──────────────────────────────────────────────────────────

    # A variations
    "AÙ": "À", "AÚ": "Á", "AÛ": "Ả", "AÜ": "Ã", "AÏ": "Ạ",
    "AÀ": "Ầ", "AÁ": "Ấ", "AÂ": "Ẩ", "AÃ": "Ẫ", "AÄ": "Ậ",
    "AÊ": "Ắ", "AË": "Ặ", "AÈ": "Ằ", "AÉ": "Ẳ", "AÌ": "Ẵ",
    "AÍ": "Ặ",
    "AÔ": "Â", "AÕ": "Ấ",
    "Aø": "À", "Aá": "Á", "Aû": "Ả", "Aõ": "Ã", "Aï": "Ạ",

    # E variations
    "EÙ": "È", "EÚ": "É", "EÛ": "Ẻ", "EÜ": "Ẽ", "EÏ": "Ẹ",
    "EÀ": "Ề", "EÁ": "Ế", "EÂ": "Ể", "EÃ": "Ễ", "EÄ": "Ệ",
    "EÂ": "Ê",

    # I variations
    "IÙ": "Ì", "IÚ": "Í", "IÛ": "Ỉ", "IÜ": "Ĩ", "IÏ": "Ị",

    # O variations
    "OÙ": "Ò", "OÚ": "Ó", "OÛ": "Ỏ", "OÜ": "Õ", "OÏ": "Ọ",
    "OÀ": "Ồ", "OÁ": "Ố", "OÂ": "Ổ", "OÃ": "Ỗ", "OÄ": "Ộ",
    "OÂ": "Ô",
    "ÔØ": "Ờ", "ÔÙ": "Ớ", "ÔÛ": "Ở", "ÔÜ": "Õ",  "ÔÏ": "Ợ", "ÔÚ": "Ớ",
    "ÔÕ": "Ỡ",

    # U variations
    "UÙ": "Ù", "UÚ": "Ú", "UÛ": "Ủ", "UÜ": "Ũ", "UÏ": "Ụ",
    "ÖØ": "Ừ", "ÖÙ": "Ứ", "ÖÛ": "Ử", "ÖÜ": "Ữ", "ÖÏ": "Ự",
    "ÖÚ": "Ứ",

    # Y variations
    "YÙ": "Ỳ", "YÚ": "Ý", "YÛ": "Ỷ", "YÜ": "Ỹ", "YÏ": "Ỵ",

    # Standalone uppercase specials
    "Ñ":  "Đ",
    "Ö":  "Ư",
    "Ô":  "Ô",   # keep as-is if already correct
    "AÊ": "Ắ",

    # ── Lowercase ──────────────────────────────────────────────────────────

    # a variations
    "aø": "à", "aá": "á", "aû": "ả", "aõ": "ã", "aï": "ạ",
    "aà": "ầ", "aá": "ấ", "aâ": "ẩ", "aã": "ẫ", "aä": "ậ",
    "aê": "ắ", "aë": "ặ", "aè": "ằ", "aé": "ẳ", "aì": "ẵ",
    "aí": "ặ",
    "aâ": "â",

    # e variations
    "eø": "è", "eá": "é", "eû": "ẻ", "eõ": "ẽ", "eï": "ẹ",
    "eà": "ề", "eá": "ế", "eâ": "ể", "eã": "ễ", "eä": "ệ",
    "eâ": "ê",

    # i variations
    "iø": "ì", "iá": "í", "iû": "ỉ", "iõ": "ĩ", "iï": "ị",

    # o variations
    "oø": "ò", "oá": "ó", "oû": "ỏ", "oõ": "õ", "oï": "ọ",
    "oà": "ồ", "oá": "ố", "oâ": "ổ", "oã": "ỗ", "oä": "ộ",
    "oâ": "ô",
    "ôø": "ờ", "ôù": "ớ", "ôû": "ở", "ôõ": "ỡ", "ôï": "ợ",

    # u variations
    "uø": "ù", "uá": "ú", "uû": "ủ", "uõ": "ũ", "uï": "ụ",
    "öø": "ừ", "öù": "ứ", "öû": "ử", "öõ": "ữ", "öï": "ự",

    # y variations
    "yø": "ỳ", "yá": "ý", "yû": "ỷ", "yõ": "ỹ", "yï": "ỵ",

    # Standalone lowercase specials
    "ñ":  "đ",
    "ö":  "ư",
    "ø":  "ò",  # fallback

    # ── Common whole-word VNI garbled patterns (high-priority) ─────────────
    # These are two-char sequences that appear very frequently and need
    # priority replacement before single-char fallbacks kick in.
    "TÖÛ": "TỬ",   "töû": "tử",
    "TOÅNG": "TỔNG", "toång": "tổng",
    "HÔÏP": "HỢP",  "hôïp": "hợp",
    "VIEÄT": "VIỆT", "vieät": "việt",
    "NAÊM": "NĂM",  "naêm": "năm",
    "NÖÔÙC": "NƯỚC", "nöôùc": "nước",
    "ÑÖÔØNG": "ĐƯỜNG","ñöôøng": "đường",
    "NGÖÔØI": "NGƯỜI","ngöôøi": "người",
    "ÑAÁT": "ĐẤT",  "ñaát": "đất",
    "THAØNH": "THÀNH","thaønh": "thành",
    "PHAÙT": "PHÁT", "phaùt": "phát",
    "ÑAÂY": "ĐÂY",  "ñaây": "đây",
    "CAÀN": "CẦN",  "caàn": "cần",
    "COÂNG": "CÔNG", "coâng": "công",
    "CUÕNG": "CŨNG", "cuõng": "cũng",
    "ÑEÀU": "ĐỀU",  "ñeàu": "đều",
    "HOÏC": "HỌC",  "hoïc": "học",
    "KHOÂNG": "KHÔNG","khoâng": "không",
    "LAØM": "LÀM",  "laøm": "làm",
    "MÖÔØI": "MƯỜI", "möôøi": "mười",
    "NHÖÕNG": "NHỮNG","nhöõng": "những",
    "CUÛA": "CỦA",  "cuûa": "của",
    "ÑOÙ": "ĐÓ",   "ñoù": "đó",
    "MOÄT": "MỘT",  "moät": "một",
    "VÔÙI": "VỚI",  "vôùi": "với",
    "THEÁ": "THẾ",  "theá": "thế",
    "THAÁT": "THẤT", "thaát": "thất",
    "BAÛN": "BẢN",  "baûn": "bản",
    "TAÁT": "TẤT",  "taát": "tất",
    "ÑÖÔÏC": "ĐƯỢC", "ñöôïc": "được",
    "HAØNH": "HÀNH", "haønh": "hành",
    "SÖÙC": "SỨC",  "söùc": "sức",
    "LÖÔÏNG": "LƯỢNG","löôïng": "lượng",
    "THÖÔØNG": "THƯỜNG","thöôøng": "thường",
    "PHAÀN": "PHẦN", "phaàn": "phần",
    "ÑÒNH": "ĐỊNH", "ñònh": "định",
    "HIEÄU": "HIỆU", "hieäu": "hiệu",
    "SAÙCH": "SÁCH", "saùch": "sách",
    "TIEÁNG": "TIẾNG","tieáng": "tiếng",
    "VIEÄC": "VIỆC", "vieäc": "việc",
    "HÖÔÙNG": "HƯỚNG","höôùng": "hướng",
    "NGAØY": "NGÀY", "ngaøy": "ngày",
    "CAÙC": "CÁC",  "caùc": "các",
    "ÑAÏI": "ĐẠI",  "ñaïi": "đại",
    "HAÏT": "HẠT",  "haït": "hạt",
    "NAØY": "NÀY",  "naøy": "này",
    "THEÅ": "THỂ",  "theå": "thể",
}

# Sort by length descending so longer patterns are replaced first
# (prevents partial replacements breaking longer matches)
SORTED_VNI_KEYS = sorted(VNI_TO_UNICODE_MAP.keys(), key=len, reverse=True)


# =============================================================================
# STEP 2: Core VNI → Unicode Conversion Function
# =============================================================================

def convert_vni_to_unicode(text: str) -> str:
    """
    Convert a VNI-Windows encoded string (misread as Latin-1) to proper UTF-8 Unicode.

    Strategy:
    1. Apply longest-match replacements first using the sorted map.
    2. Fall back to byte-level decoding for any remaining garbled characters.

    Args:
        text: Raw string extracted from a VNI-encoded PDF.

    Returns:
        Clean UTF-8 Vietnamese string.
    """
    if not text:
        return text

    # Pass 1: Replace known multi-char and single-char VNI patterns
    for vni_pattern in SORTED_VNI_KEYS:
        text = text.replace(vni_pattern, VNI_TO_UNICODE_MAP[vni_pattern])

    # Pass 2: Byte-level fallback for any remaining high-byte sequences
    # Re-encode to latin-1 bytes and walk through the VNI pair logic
    try:
        text = _byte_level_vni_decode(text)
    except Exception:
        pass  # If byte-level fails, keep string-level result

    return text


def _byte_level_vni_decode(text: str) -> str:
    """
    Byte-level VNI decoder as a fallback.
    Handles cases not covered by the string map.
    """
    try:
        data = text.encode('latin-1', errors='replace')
    except Exception:
        return text

    pair_map = {
        # lowercase
        ('a', 0xF8): 'à', ('a', 0xF9): 'á', ('a', 0xFB): 'ả', ('a', 0xF5): 'ã', ('a', 0xEF): 'ạ',
        ('a', 0xE0): 'ầ', ('a', 0xE1): 'ấ', ('a', 0xE2): 'ẩ', ('a', 0xE3): 'ẫ', ('a', 0xE4): 'ậ',
        ('a', 0xE2): 'â',
        ('e', 0xF8): 'è', ('e', 0xF9): 'é', ('e', 0xFB): 'ẻ', ('e', 0xF5): 'ẽ', ('e', 0xEF): 'ẹ',
        ('e', 0xE0): 'ề', ('e', 0xE1): 'ế', ('e', 0xE2): 'ể', ('e', 0xE3): 'ễ', ('e', 0xE4): 'ệ',
        ('i', 0xF8): 'ì', ('i', 0xF9): 'í', ('i', 0xFB): 'ỉ', ('i', 0xF5): 'ĩ', ('i', 0xEF): 'ị',
        ('o', 0xF8): 'ò', ('o', 0xF9): 'ó', ('o', 0xFB): 'ỏ', ('o', 0xF5): 'õ', ('o', 0xEF): 'ọ',
        ('o', 0xE0): 'ồ', ('o', 0xE1): 'ố', ('o', 0xE2): 'ổ', ('o', 0xE3): 'ỗ', ('o', 0xE4): 'ộ',
        ('u', 0xF8): 'ù', ('u', 0xF9): 'ú', ('u', 0xFB): 'ủ', ('u', 0xF5): 'ũ', ('u', 0xEF): 'ụ',
        ('y', 0xF8): 'ỳ', ('y', 0xF9): 'ý', ('y', 0xFB): 'ỷ', ('y', 0xF5): 'ỹ', ('y', 0xEF): 'ỵ',
        ('ö', 0xF8): 'ừ', ('ö', 0xF9): 'ứ', ('ö', 0xFB): 'ử', ('ö', 0xF5): 'ữ', ('ö', 0xEF): 'ự',
        ('ô', 0xF8): 'ờ', ('ô', 0xF9): 'ớ', ('ô', 0xFB): 'ở', ('ô', 0xF5): 'ỡ', ('ô', 0xEF): 'ợ',
    }

    standalone = {
        0xD1: 'Đ',  # Ñ → Đ (uppercase)
        0xF1: 'đ',  # ñ → đ (lowercase)
        0xD6: 'Ư',  # Ö → Ư (uppercase)
        0xF6: 'ư',  # ö → ư (lowercase)
    }

    result = []
    for b in data:
        if b < 128:
            result.append(chr(b))
            continue

        handled = False
        if result:
            last = result[-1]
            key = (last.lower(), b)
            if key in pair_map:
                combined = pair_map[key]
                result[-1] = combined.upper() if last.isupper() else combined
                handled = True

        if not handled:
            result.append(standalone.get(b, chr(b)))

    return ''.join(result)


# =============================================================================
# STEP 3: Text Cleaning
# =============================================================================

def clean_text(text: str) -> str:
    """
    Clean extracted PDF text:
    - Normalize line breaks
    - Remove excessive whitespace
    - Remove PDF artifacts (page numbers, headers, etc.)
    - Strip leading/trailing whitespace

    Args:
        text: Raw text after VNI conversion.

    Returns:
        Cleaned text string.
    """
    # Normalize Windows line endings
    text = text.replace('\r\n', '\n').replace('\r', '\n')

    # Remove hyphenated line breaks (common in PDFs): "từ-\nngữ" → "từngữ"
    text = re.sub(r'-\n', '', text)

    # Collapse multiple blank lines into a single blank line
    text = re.sub(r'\n{3,}', '\n\n', text)

    # Remove lines that are purely numeric (page numbers)
    text = re.sub(r'^\s*\d+\s*$', '', text, flags=re.MULTILINE)

    # Collapse multiple spaces/tabs into a single space
    text = re.sub(r'[ \t]+', ' ', text)

    # Strip each line
    lines = [line.strip() for line in text.split('\n')]
    text = '\n'.join(lines)

    # Final strip
    return text.strip()


# =============================================================================
# STEP 4: PDF Text Extraction
# =============================================================================

def extract_text_from_pdf(pdf_path: str) -> str:
    """
    Extract raw text from a PDF file using pdfplumber.
    pdfplumber handles complex layouts better than PyPDF2.

    Args:
        pdf_path: Absolute or relative path to the PDF file.

    Returns:
        Concatenated raw text from all pages.
    """
    all_text = []

    with pdfplumber.open(pdf_path) as pdf:
        total_pages = len(pdf.pages)
        print(f"[INFO] Extracting text from {total_pages} pages: {pdf_path}")

        for i, page in enumerate(pdf.pages):
            page_text = page.extract_text()
            if page_text:
                all_text.append(page_text)
            else:
                print(f"[WARN] Page {i + 1} returned no text (may be image-based).")

    return "\n".join(all_text)


# =============================================================================
# STEP 5: Chunking with LangChain
# =============================================================================

def chunk_text(
    text: str,
    source: str = "unknown",
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Split clean Unicode text into LangChain Document chunks ready for embedding.

    Args:
        text:          Clean UTF-8 Vietnamese text.
        source:        Filename or identifier to attach as metadata.
        chunk_size:    Max characters per chunk.
        chunk_overlap: Overlap between consecutive chunks (for context continuity).

    Returns:
        List of LangChain Document objects with text + metadata.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        # Vietnamese-aware separators — paragraph > sentence > word
        separators=["\n\n", "\n", "。", ".", "!", "?", "،", " ", ""],
        length_function=len,
    )

    chunks = splitter.create_documents(
        texts=[text],
        metadatas=[{"source": source}],
    )

    print(f"[INFO] Created {len(chunks)} chunks from '{source}'")
    return chunks


# =============================================================================
# MAIN PIPELINE — plug this into your RAG ingestion flow
# =============================================================================

def process_vni_pdf(
    pdf_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
) -> list[Document]:
    """
    Full pipeline: VNI PDF → clean UTF-8 Unicode → LangChain Document chunks.

    Args:
        pdf_path:      Path to the VNI-encoded PDF file.
        chunk_size:    Characters per chunk (default 1000).
        chunk_overlap: Overlap between chunks (default 200).

    Returns:
        List of LangChain Document chunks ready for Gemini embedding.

    Usage:
        chunks = process_vni_pdf("tu_vi.pdf")
        # Then pass to your vector store:
        # vectorstore.add_documents(chunks)
    """
    print(f"\n{'='*60}")
    print(f"Processing: {pdf_path}")
    print(f"{'='*60}")

    # Step 1: Extract raw text from PDF
    raw_text = extract_text_from_pdf(pdf_path)
    print(f"[INFO] Extracted {len(raw_text):,} raw characters")

    # Step 2: Convert VNI → Unicode
    unicode_text = convert_vni_to_unicode(raw_text)
    print(f"[INFO] VNI → Unicode conversion complete")

    # Step 3: Clean the text
    clean = clean_text(unicode_text)
    print(f"[INFO] Cleaned text: {len(clean):,} characters")

    # Step 4: Chunk for embedding
    chunks = chunk_text(
        text=clean,
        source=pdf_path,
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )

    # Preview first chunk
    if chunks:
        print(f"\n[PREVIEW] First chunk:\n{'-'*40}")
        print(chunks[0].page_content[:300])
        print(f"{'-'*40}\n")

    return chunks


# =============================================================================
# EXAMPLE — run directly to test
# =============================================================================

if __name__ == "__main__":
    import sys

    # Test with a quick string conversion first
    test_input = "TÖÛ VI TOÅNG HÔÏP - KHOÂNG GIAN VIEÄT NÖÔÙC"
    converted = convert_vni_to_unicode(test_input)
    print(f"[TEST] VNI → Unicode:")
    print(f"  Input:  {test_input}")
    print(f"  Output: {converted}")
    print()

    # Process a real PDF if path is provided
    if len(sys.argv) > 1:
        pdf_file = sys.argv[1]
        chunks = process_vni_pdf(pdf_file)
        print(f"[DONE] {len(chunks)} chunks ready for embedding.")
    else:
        print("[USAGE] python vni_rag_pipeline.py path/to/your/file.pdf")
