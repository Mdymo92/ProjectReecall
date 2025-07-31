import os
import json
import logging
import typer
import re
import unicodedata
import spacy
from functools import lru_cache
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)
app = typer.Typer()

# Language -> spaCy model mapping
MODEL_MAP = {
    'fr': 'fr_core_news_sm',
    'de': 'de_core_news_sm',
    'en': 'en_core_web_sm',
    'es': 'es_core_news_sm'
}

@lru_cache(maxsize=4)
def get_tokenizer(lang: str):
    """Load and return the spaCy tokenizer for the given language."""
    model = MODEL_MAP.get(lang)
    if not model:
        logger.warning(f"No spaCy model found for '{lang}', skipping tokenization.")
        return None
    try:
        return spacy.load(model)
    except Exception as e:
        logger.error(f"Error loading model {model}: {e}")
        return None

def load_checkpoint(checkpoint_path: Path) -> int:
    if checkpoint_path.exists():
        try:
            return int(checkpoint_path.read_text().strip())
        except Exception:
            logger.warning("Invalid checkpoint, restarting from 0.")
    return 0

def save_checkpoint(checkpoint_path: Path, idx: int):
    checkpoint_path.write_text(str(idx))

def collapse_phrases(text: str) -> str:
    """
    Remove repeated adjacent sequences of words.
    e.g. "I am here I am here" -> "I am here"
    """
    pattern = re.compile(r'\b((?:\w+\s+){2,}?)\1', re.IGNORECASE)
    while True:
        new_text = pattern.sub(r'\1', text)
        if new_text == text:
            break
        text = new_text
    return text

def clean_text(text: str) -> str:
    # 1) Normalize Unicode -> ASCII
    text = unicodedata.normalize('NFKD', text)
    text = text.encode('ascii', 'ignore').decode('ascii')
    # 2) Remove placeholders
    text = re.sub(r'PHONE_NUMBER_\d+', '', text)
    # 3) Remove punctuation
    text = re.sub(r"[^\w\d\s]", ' ', text)
    # 4) Separate letters/numbers and isolate numbers
    text = re.sub(r"(?<=\D)(?=\d)|(?<=\d)(?=\D)", ' ', text)
    text = re.sub(r"(\d+)", r" \1 ", text)
    # 5) Remove repeated numbers
    text = re.sub(r"\b(\d+)( \1\b)+", r"\1", text)
    # 6) Remove repeated words
    text = re.sub(r"\b(\w+)( \1\b)+", r"\1", text)
    # 7) Collapse repeated phrases
    text = collapse_phrases(text)
    # 8) Lowercase
    text = text.lower()
    # 9) Collapse multiple spaces
    text = re.sub(r"\s+", ' ', text).strip()
    return text

@app.command()
def preprocess(
    input_dir: str = typer.Option(..., help="Directory containing intermediate JSONC files"),
    output_dir: str = typer.Option(..., help="Directory for cleaned output files"),
    batch_size: int = typer.Option(100, help="Batch size"),
    checkpoint_file: str = typer.Option('.checkpoint', help="Checkpoint file"),
):
    """Clean messages and apply multilingual segmentation with batching and checkpointing."""
    in_path = Path(input_dir)
    out_path = Path(output_dir)
    cp_path = Path(checkpoint_file)
    os.makedirs(out_path, exist_ok=True)

    files = sorted([f for f in in_path.iterdir() if f.suffix in ('.jsonc', '.json')])
    start_idx = load_checkpoint(cp_path)
    total = len(files)
    logger.info(f"Debug: {total} files found in {input_dir}")
    logger.info(f"Resuming from batch index: {start_idx}")

    for idx in range(start_idx, total, batch_size):
        batch = files[idx: idx + batch_size]
        logger.info(f"Processing batch {idx} to {idx + len(batch) - 1} out of {total - 1}")
        for file in batch:
            try:
                data = json.loads(file.read_text(encoding='utf-8'))
                lang = data.get('language', 'fr')
                tokenizer = get_tokenizer(lang)
                new_messages = []
                for msg in data.get('messages', []):
                    cleaned = clean_text(msg.get('text', ''))
                    # Only keep non-empty cleaned messages
                    if cleaned:
                        if tokenizer:
                            tokens = [token.text for token in tokenizer(cleaned)]
                            msg['text'] = ' '.join(tokens)
                        else:
                            msg['text'] = cleaned
                        new_messages.append(msg)
                data['messages'] = new_messages
                out_file = out_path / file.name
                out_file.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
            except Exception as e:
                logger.error(f"Error processing {file.name}: {e}")
        save_checkpoint(cp_path, idx + len(batch))
        logger.info(f"Batch completed, checkpoint saved: {idx + len(batch)}")

    typer.echo("Preprocessing complete.")

if __name__ == "__main__":
    app()
