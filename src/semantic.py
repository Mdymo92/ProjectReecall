import json
import logging
import os
import glob
from typing import List, Dict, Any
from dotenv import load_dotenv
import typer
from tqdm import tqdm
import openai
from openai import OpenAIError, RateLimitError

app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

load_dotenv()  # Load environment variables from .env file
# Use OpenAI API with GPT-3.5 model
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The environment variable OPENAI_API_KEY is missing. Check your .env file.")
# Path to the local cache
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

def label_utterance(text: str) -> Dict[str, Any]:
    if text in label_cache:
        return label_cache[text]

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "You are an assistant that classifies utterances by themes and categories."},
                {"role": "user", "content": f"Text: '{text}'\nRespond ONLY with a JSON containing the fields theme, category, confidence."}
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
        logger.error("OpenAI error for text '%s': %s", text, e)
        label = {"theme": "unknown", "category": "unknown", "confidence": 0.0}

    label_cache[text] = label
    save_cache()
    return label

def label_conversation(conv: Dict[str, Any]) -> List[Dict[str, Any]]:
    cid = conv.get("conversation_id", "")
    labels: List[Dict[str, Any]] = []
    for msg in conv.get("messages", []):
        text = msg.get("text", "")
        lbl = label_utterance(text)
        labels.append({
            "conversation_id": cid,
            "speaker": msg.get("speaker", ""),
            **lbl
        })
    return labels

@app.command()
def batch_label(
    input_dir: str = typer.Argument(..., help="Folder containing pre-processed JSON/JSONC files"),
    output_path: str = typer.Argument(..., help="Output JSONL file with one label per line"),
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
                for lbl in label_conversation(data):
                    fout.write(json.dumps(lbl, ensure_ascii=False) + "\n")
            except Exception:
                logger.exception("Error processing %s", path)

    logger.info("Labels written to %s", output_path)

if __name__ == "__main__":
    app()
