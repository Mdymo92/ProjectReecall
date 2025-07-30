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
    input_dir: str = typer.Option(..., help="Dossier contenant les fichiers JSON de transcription bruts"),
    output_dir: str = typer.Option(..., help="Dossier de sortie pour JSONC intermédiaires")
):
    """Ingest raw JSON transcriptions and generate intermediary JSONC files."""
    os.makedirs(output_dir, exist_ok=True)
    files = [f for f in os.listdir(input_dir) if f.endswith('-ANON.txt') or f.endswith('.txt')]
    for fname in files:
        in_path = Path(input_dir) / fname
        # Extraire le role depuis le nom de fichier (FR_Senior_... ou DE_Junior_...)
        parts = fname.split('_')
        role = parts[1] if len(parts) > 1 else 'unknown'
        try:
            logger.info(f"Reading file {fname}")
            with open(in_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load {fname}: {e}")
            continue

        transcription = data.get('transcription', {})
        call_id = transcription.get('call_id')
        if call_id is None:
            logger.warning(f"No call_id in {fname}, skipping")
            continue

        utterances = transcription.get('content', {}).get('utterances', [])
        messages = []
        for utt in utterances:
            speaker = 'agent' if utt.get('participant_type') == 'internal' else 'client'
            msg = {
                'speaker': speaker,
                'text': utt.get('text', '').strip(),
                'start_time': utt.get('start_time'),
                'end_time': utt.get('end_time')
            }
            messages.append(msg)

        obj = {
            'conversation_id': str(call_id),
            'role': role,
            'messages': messages
        }

        out_path = Path(output_dir) / f"{call_id}.jsonc"
        try:
            with open(out_path, 'w', encoding='utf-8') as out_f:
                json.dump(obj, out_f, ensure_ascii=False, indent=2)
            logger.info(f"Wrote intermediary JSONC to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write {out_path}: {e}")

    typer.echo("Ingestion complete.")

if __name__ == "__main__":
    app()
