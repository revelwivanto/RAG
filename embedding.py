import os
import json
from pathlib import Path

from dotenv import load_dotenv
from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.node_parser import SentenceSplitter
from llama_index.vector_stores.qdrant import QdrantVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import qdrant_client

load_dotenv()

google_api_key = 
qdrant_url = 
qdrant_api_key = 

Settings.embed_model = HuggingFaceEmbedding(
    model_name="BAAI/bge-large-en-v1.5"  # 1024-dim, better quality, slower
)

client = qdrant_client.QdrantClient(
    url=qdrant_url,
    api_key=qdrant_api_key,
    # The cluster is in sa-east-1; the 5s default is not enough for a
    # round trip with 1024-dim vectors and the writes time out.
    timeout=120,
    check_compatibility=False,
)
vector_store = QdrantVectorStore(
    client=client,
    collection_name="bni_training",
    batch_size=8,
)
storage_context = StorageContext.from_defaults(vector_store=vector_store)

def _page_documents(json_path, pages):
    """parse.py output: a list of page objects with text/table blocks."""
    documents = []

    for page in pages:
        content = [block["text"] for block in page.get("text_blocks", [])]
        content.extend(
            "\n".join(" | ".join(row) for row in table["data"])
            for table in page.get("table_blocks", [])
        )

        if content:
            documents.append(Document(
                text="\n".join(content),
                metadata={
                    "source": json_path.name,
                    "page": page["page"],
                },
            ))

    return documents


def _flat_document(json_path, payload):
    """Flat {"text": "..."} output -- one blob, no page structure to keep."""
    text = payload.get("text", "").strip()
    if not text:
        return []

    return [Document(text=text, metadata={"source": json_path.name})]


def load_parsed_documents(parsed_dir):
    documents = []

    for json_path in sorted(Path(parsed_dir).rglob("*.json")):
        payload = json.loads(json_path.read_text(encoding="utf-8"))

        if isinstance(payload, list):
            found = _page_documents(json_path, payload)
        elif isinstance(payload, dict) and "text" in payload:
            found = _flat_document(json_path, payload)
        else:
            print(f"  {json_path.name}: skipped, unrecognised shape ({type(payload).__name__})")
            continue

        print(f"  {json_path.name}: {len(found)} document(s)")
        documents.extend(found)

    return documents


data_candidates = [
    Path("/kaggle/input/datasets/revelelel/parsed-result"),
    Path("/kaggle/input/parsed-result"),
    Path.cwd() / "parsed_result",
    Path.cwd() / "data" / "parsed_result",
]
parsed_dir = next((path for path in data_candidates if path.is_dir()), None)
if parsed_dir is None:
    raise FileNotFoundError(
        "No parsed-result directory found. Checked: "
        + ", ".join(str(path) for path in data_candidates)
    )

documents = load_parsed_documents(parsed_dir)
if not documents:
    raise RuntimeError(f"No JSON pages found in {parsed_dir}")

nodes = SentenceSplitter(chunk_size=512, chunk_overlap=50).get_nodes_from_documents(documents)
print(f"Loaded {len(documents)} pages and created {len(nodes)} embedding chunks.")

VectorStoreIndex(nodes, storage_context=storage_context)

print("Qdrant collections:", [item.name for item in client.get_collections().collections])

print("Done. Embeddings are stored in the bni_training collection.")
