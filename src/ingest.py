import os
import json
import logging
from dotenv import load_dotenv
import typer
from pathlib import Path

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app = typer.Typer()

@app.command()
def ingest(
    input_dir: str = typer.Option(..., help="Directory containing raw transcription JSON files"),
    output_dir: str = typer.Option(..., help="Output directory for intermediate JSONC files")
):
    """Ingest raw JSON transcriptions and generate intermediary JSONC files."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('-ANON.txt') or f.endswith('.txt')]
    for fname in files:
        in_path = Path(input_dir) / fname
        # Extract the role from the filename (e.g., FR_Senior_... or DE_Junior_...)
        parts = fname.split('_')
        role = parts[1] if len(parts) > 1 else 'unknown'
        try:
            logger.info(f"Reading file {fname}")
            data = json.loads(in_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Failed to load {fname}: {e}")
            continue

        transcription = data.get('transcription', {})
        call_id = transcription.get('call_id')
        if call_id is None:
            logger.warning(f"No call_id in {fname}, skipping")
            continue

        # Extract the language
        language = transcription.get('content', {}).get('language', 'unknown')

        utterances = transcription.get('content', {}).get('utterances', [])
        messages = []
        for utt in utterances:
            speaker = 'agent' if utt.get('participant_type') == 'internal' else 'client'
            messages.append({
                'speaker': speaker,
                'text': utt.get('text', '').strip(),
                'start_time': utt.get('start_time'),
                'end_time': utt.get('end_time')
            })

        obj = {
            'conversation_id': str(call_id),
            'role': role,
            'language': language,
            'messages': messages
        }

        out_path = Path(output_dir) / f"{call_id}.jsonc"
        try:
            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            logger.info(f"Wrote intermediary JSONC to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write {out_path}: {e}")

    typer.echo("Ingestion complete.")

if __name__ == "__main__":
    app()
