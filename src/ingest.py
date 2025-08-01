import os
import json
import logging
import hashlib
from dotenv import load_dotenv
import typer
from pathlib import Path

# Load environment variables from .env file
load_dotenv()

# Configure logging format and level
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

# Typer command-line application
app = typer.Typer()

def anonymize_phone(phone: str) -> str:
    """Hash phone numbers using SHA256 for GDPR compliance."""
    return hashlib.sha256(phone.encode('utf-8')).hexdigest()

@app.command()
def ingest(
    input_dir: str = typer.Option(..., help="Directory containing raw transcription JSON files"),
    output_dir: str = typer.Option(..., help="Output directory for intermediate JSONC files")
):
    """Ingest raw JSON transcriptions and generate intermediary JSONC files."""

    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Select all text files ending with -ANON.txt or .txt
    files = [f for f in os.listdir(input_dir) if f.endswith('-ANON.txt') or f.endswith('.txt')]

    for fname in files:
        in_path = Path(input_dir) / fname
        parts = fname.split('_')
        role = parts[1] if len(parts) > 1 else 'unknown'

        try:
            logger.info(f"Reading file {fname}")
            data = json.loads(in_path.read_text(encoding='utf-8'))
        except Exception as e:
            logger.error(f"Failed to load {fname}: {e}")
            continue

        # Extract transcription object
        trx = data.get('transcription', {})
        call_id = trx.get('call_id')
        if call_id is None:
            logger.warning(f"No call_id in {fname}, skipping")
            continue

        # Additional metadata
        call_created = trx.get('call_created_at')
        language     = trx.get('content', {}).get('language', 'unknown')

        # Extract and process each utterance
        messages = []
        for utt in trx.get('content', {}).get('utterances', []):
            speaker = 'agent' if utt.get('participant_type') == 'internal' else 'client'
            phone   = utt.get('phone_number')
            user_id = utt.get('user_id')
            messages.append({
                'speaker':      speaker,
                'text':         utt.get('text', '').strip(),
                'start_time':   utt.get('start_time'),
                'end_time':     utt.get('end_time'),
                'user_id':      user_id,
                'phone_hash':   anonymize_phone(phone) if phone else None  # Hash the phone if available
            })

        # Construct the intermediate JSON object
        obj = {
            'conversation_id': str(call_id),
            'call_created_at': call_created,
            'role':            role,
            'language':        language,
            'messages':        messages
        }

        # Output file path
        out_path = Path(output_dir) / f"{call_id}.jsonc"
        try:
            out_path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding='utf-8')
            logger.info(f"Wrote intermediary JSONC to {out_path}")
        except Exception as e:
            logger.error(f"Failed to write {out_path}: {e}")

    # Final message once all files are processed
    typer.echo("Ingestion complete.")

if __name__ == "__main__":
    app()
