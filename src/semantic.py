import json
import logging
import os
import glob
from typing import List, Dict, Any
import typer
from tqdm import tqdm
import openai
from openai import OpenAIError, RateLimitError
from langdetect import detect
from dotenv import load_dotenv


app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()  # Load variables from .env
# Use OpenAI API with GPT-3.5 model
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The environment variable OPENAI_API_KEY is missing. Check your .env file.")

# Path to local cache
CACHE_PATH = "./label_cache.json"

# Load or initialize the cache
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as cf:
        try:
            label_cache = json.load(cf)
        except json.JSONDecodeError:
            label_cache = {}
else:
    label_cache = {}

def save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as cf:
        json.dump(label_cache, cf, ensure_ascii=False, indent=2)

def extract_use_cases(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    cid = conversation.get("conversation_id", "")
    turns = conversation.get("messages", [])

    dialogue = "\n".join([f"{m.get('role', '')}: {m.get('text', '')}" for m in turns if m.get("text")])

    prompt = (
        "Here is a conversation between a customer and an agent."
        " Analyze the different exchanges to extract use cases in the form of needs and solutions."
        " Provide a JSON list of dictionaries with two keys: 'need' (problem expressed by the customer) and 'solution' (response or resolution proposed by the agent)."
        " Translate the conversation if necessary and respond only in French."
        " Example: [\n  {\"need\": \"I can’t pay with my card.\", \"solution\": \"Try another card or restart the app.\"}\n]"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Conversation:\n{dialogue}"}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        return json.loads(content)
    except Exception as e:
        logger.error("Error extracting use cases: %s", e)
        return []

def label_conversation_summary(conversation: Dict[str, Any]) -> Dict[str, Any]:
    cid = conversation.get("conversation_id", "")
    texts = [msg.get("text", "") for msg in conversation.get("messages", []) if msg.get("text")]
    full_text = "\n".join(texts).strip()

    if not full_text:
        return {"conversation_id": cid, "theme": "unknown", "category": "unknown", "confidence": 0.0}

    if full_text in label_cache:
        return {"conversation_id": cid, **label_cache[full_text]}

    # Improved prompt
    system_prompt = (
        "You are an assistant who analyzes a full conversation and identifies the main theme and the corresponding category."
        " You must generalize recurring categories and themes into a standardized format."
        " First translate the conversation into French if it is in a foreign language (English, Spanish, German...)."
        " Use a standardized taxonomy. Respond ONLY in JSON with the following structure:"
        "\n\n"
        "Example 1:\n"
        "{\n  \"theme\": \"Payment\",\n  \"category\": \"Card declined\",\n  \"confidence\": 0.95\n}\n"
        "\n"
        "Example 2:\n"
        "{\n  \"theme\": \"Technical issue\",\n  \"category\": \"Barrier won’t open\",\n  \"confidence\": 0.91\n}\n"
        "\n"
        "Possible list of themes: ['Payment', 'Login', 'Access', 'Booking', 'Exit', 'Account', 'Information']\n"
        "Possible list of categories: ['Card declined', 'Forgot password', 'Invalid code', 'Barrier stuck', 'Booking error', 'Account blocked', 'Balance request']\n"
        "The answer must always be in French."
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": f"Conversation:\n{full_text}"}
            ],
            temperature=0.0
        )
        content = response.choices[0].message.content.strip()
        result = json.loads(content)
        label = {
            "theme": result.get("theme", "unknown"),
            "category": result.get("category", "unknown"),
            "confidence": float(result.get("confidence", 0.0))
        }
    except RateLimitError as e:
        logger.error("Quota exceeded. Details: %s", e)
        raise typer.Exit(code=2)
    except Exception as e:
        logger.error("OpenAI error for conversation '%s': %s", full_text, e)
        label = {"theme": "unknown", "category": "unknown", "confidence": 0.0}

    label_cache[full_text] = label
    save_cache()
    return {"conversation_id": cid, **label}

@app.command()
def batch_label(
    input_dir: str = typer.Argument(..., help="Folder containing pre-processed JSON/JSONC files"),
    output_path: str = typer.Argument(..., help="Output JSONL file with one label per conversation"),
    pattern: str = typer.Option("*.json,*.jsonc", help="Glob patterns, separated by commas")
):
    files: List[str] = []
    for pat in pattern.split(','):
        files.extend(glob.glob(os.path.join(input_dir, pat.strip())))
    files = sorted(set(files))

    if not files:
        logger.error("No files found for patterns %s in %s", pattern, input_dir)
        raise typer.Exit(code=1)

    with open(output_path, "w", encoding="utf-8") as fout:
        for path in tqdm(files, desc="Labeling conversations"):
            try:
                raw = open(path, "r", encoding="utf-8").read()
                data = (
                    json.loads(raw) if path.endswith(".json")
                    else json.loads("\n".join(
                        ln for ln in raw.splitlines() if not ln.strip().startswith("//")
                    ))
                )
                if isinstance(data, list):
                    for conv in data:
                        label = label_conversation_summary(conv)
                        conv_id = conv.get("conversation_id", "")
                        use_cases = extract_use_cases(conv)
                        label["use_cases"] = use_cases
                        fout.write(json.dumps(label, ensure_ascii=False) + "\n")
                elif isinstance(data, dict):
                    label = label_conversation_summary(data)
                    use_cases = extract_use_cases(data)
                    label["use_cases"] = use_cases
                    fout.write(json.dumps(label, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Error processing %s", path)

    logger.info("Labels written to %s", output_path)

if __name__ == "__main__":
    app()
