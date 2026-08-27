import os
import glob
from pydantic import BaseModel, Field
import instructor
import google.generativeai as genai
# Note: You can easily swap this for the OpenAI or Anthropic SDK

DATA_DIR = "data"
OUTPUT_DIR = "parsed_result"

# 1. Define the nested object first
class SearchSummary(BaseModel):
    specs_searched: str
    cheapest_vendor_found: str
    vendor_channel_type: str
    real_price: float
    price_savings_vs_requested: float

# 2. Define the main payload
class PurchaseRequest(BaseModel):
    request_id: str
    department: str
    item_category: str
    requested_unit_price: float
    historical_avg_price: float
    price_variance_ratio: float
    quantity: int
    total_amount: float
    vendor_risk_score: float
    dept_budget_remaining: float
    is_urgent: int # Or bool, depending on your downstream needs
    search_summary: SearchSummary

# 3. Initialize the client (Example using Instructor + Gemini API)
genai.configure(api_key="AIzaSyDpeQhtexYE4KvC_k7jaVHtsdcxYS8NEjU")
client = instructor.from_gemini(
    genai.GenerativeModel(model_name="gemini-3.1-flash-lite"),
    mode=instructor.Mode.GEMINI_JSON
)

# 4. Pass each document in data/ directly to the VLM and save its extraction
# (Assuming you are using Gemini, which natively handles PDF files)
os.makedirs(OUTPUT_DIR, exist_ok=True)

for pdf_path in glob.glob(os.path.join(DATA_DIR, "*.pdf")):
    filename = os.path.splitext(os.path.basename(pdf_path))[0]
    print(f"Parsing {pdf_path}...")

    sample_pdf = genai.upload_file(path=pdf_path)

    try:
        extraction = client.chat.completions.create(
            response_model=PurchaseRequest, # This forces the output to match your JSON
            messages=[
                {
                    "role": "user",
                    "content": [
                        sample_pdf,
                        "Extract the purchase request details from this document. Calculate the price variance ratio if not explicitly stated."
                    ]
                }
            ]
        )
    except Exception as e:
        print(f"Failed to parse {pdf_path}: {e}")
        continue

    output_path = os.path.join(OUTPUT_DIR, f"{filename}.json")
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(extraction.model_dump_json(indent=2))
    print(f"Saved {output_path}")