
from pypdf import PdfReader
from dotenv import load_dotenv
from langchain_text_splitters import (
    CharacterTextSplitter,
    RecursiveCharacterTextSplitter,
    TokenTextSplitter,
    NLTKTextSplitter,
    MarkdownHeaderTextSplitter,
    HTMLHeaderTextSplitter
)

from langchain_experimental.text_splitter import SemanticChunker
from langchain_openai import OpenAIEmbeddings
import os 

load_dotenv()
# LOAD PDF
PDF_PATH = "somatosensory.pdf"

reader = PdfReader(PDF_PATH)

text = ""

for page_num, page in enumerate(reader.pages):

    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"

print("\n")
print("=" * 100)
print("PDF LOADED")
print("=" * 100)

print(f"Total Characters : {len(text)}")


# HELPER
def analyze_chunks(splitter_name, chunks):

    print("\n")
    print("=" * 100)
    print(splitter_name)
    print("=" * 100)

    print(f"Total Chunks : {len(chunks)}")

    if len(chunks) == 0:
        return

    avg_size = sum(len(chunk) for chunk in chunks) / len(chunks)

    print(f"Average Chunk Length : {avg_size:.2f}")

    print("\nFIRST CHUNK:\n")
    print(chunks[0][:500])

    print("\n")


# 1. CHARACTER TEXT SPLITTER
# Splits based mainly on size and separators.
character_splitter = CharacterTextSplitter(
    separator="\n",
    chunk_size=1000,
    chunk_overlap=200
)

character_chunks = character_splitter.split_text(text)

analyze_chunks(
    "CharacterTextSplitter",
    character_chunks
)


# 2. RECURSIVE CHARACTER SPLITTER
# Tries to preserve paragraphs and sentences before splitting, even if the chunk becomes slightly larger than the target size.
recursive_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

recursive_chunks = recursive_splitter.split_text(text)

analyze_chunks(
    "RecursiveCharacterTextSplitter",
    recursive_chunks
)


# 3. TOKEN SPLITTER
token_splitter = TokenTextSplitter(
    chunk_size=300,
    chunk_overlap=50
)

token_chunks = token_splitter.split_text(text)

analyze_chunks(
    "TokenTextSplitter",
    token_chunks
)


# 4. SENTENCE SPLITTER (NLTK)
# sentence_splitter = NLTKTextSplitter()

# sentence_chunks = sentence_splitter.split_text(text)

# analyze_chunks(
#     "NLTKTextSplitter",
#     sentence_chunks
# )


# 5. PARAGRAPH SPLITTER
paragraph_chunks = [
    paragraph.strip()
    for paragraph in text.split("\n\n")
    if paragraph.strip()
]

analyze_chunks(
    "Paragraph Splitter",
    paragraph_chunks
)

# 6. SEMANTIC CHUNKER
try:

    semantic_splitter = SemanticChunker(
        OpenAIEmbeddings(api_key=os.getenv("OPENAI_API_KEY"))
    )

    semantic_docs = semantic_splitter.create_documents(
        [text]
    )

    semantic_chunks = [
        doc.page_content
        for doc in semantic_docs
    ]

    analyze_chunks(
        "SemanticChunker",
        semantic_chunks
    )

except Exception as e:

    print("\nSemanticChunker Failed")
    print(e)

# 7. MARKDOWN HEADER SPLITTER
markdown_text = """
# Introduction

This is introduction section.

## History

History paragraph.

## Applications

Applications paragraph.

# Conclusion

Final summary.
"""

headers = [
    ("#", "Header1"),
    ("##", "Header2")
]

markdown_splitter = MarkdownHeaderTextSplitter(
    headers_to_split_on=headers
)

markdown_docs = markdown_splitter.split_text(
    markdown_text
)

markdown_chunks = [
    doc.page_content
    for doc in markdown_docs
]

analyze_chunks(
    "MarkdownHeaderTextSplitter",
    markdown_chunks
)


# 8. HTML HEADER SPLITTER
html_text = """
<html>

<h1>Introduction</h1>

<p>Intro content.</p>

<h2>History</h2>

<p>History content.</p>

<h2>Applications</h2>

<p>Applications content.</p>

</html>
"""

html_splitter = HTMLHeaderTextSplitter(
    headers_to_split_on=[
        ("h1", "Header1"),
        ("h2", "Header2")
    ]
)

html_docs = html_splitter.split_text(
    html_text
)

html_chunks = [
    doc.page_content
    for doc in html_docs
]

analyze_chunks(
    "HTMLHeaderTextSplitter",
    html_chunks
)

