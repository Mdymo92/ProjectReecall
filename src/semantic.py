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

# Initialize the CLI app
app = typer.Typer()
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Load environment variables from a .env file
load_dotenv()

# Retrieve OpenAI API key from environment
openai.api_key = os.getenv("OPENAI_API_KEY")
if openai.api_key is None:
    raise ValueError("The environment variable OPENAI_API_KEY is missing. Check your .env file.")

# Define local cache path for conversation labels
CACHE_PATH = "./label_cache.json"

# Load cache if it exists, otherwise initialize it
if os.path.exists(CACHE_PATH):
    with open(CACHE_PATH, "r", encoding="utf-8") as cf:
        try:
            label_cache = json.load(cf)
        except json.JSONDecodeError:
            label_cache = {}
else:
    label_cache = {}

# Save updated cache to disk
def save_cache():
    with open(CACHE_PATH, "w", encoding="utf-8") as cf:
        json.dump(label_cache, cf, ensure_ascii=False, indent=2)

# Translate non-French text to French using OpenAI
def translate_to_french(text: str) -> str:
    if detect(text) == 'fr':
        return text
    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": "Tu es un assistant de traduction. Traduis uniquement en français, sans autre texte ni commentaire."},
                {"role": "user", "content": f"Traduis ceci : {text}"}
            ],
            temperature=0.0
        )
        translated = response.choices[0].message.content.strip()
        return translated
    except Exception as e:
        logger.warning(f"❌ Traduction échouée pour '{text}' : {e}")
        return text

# Extract use cases from a conversation using GPT
def extract_use_cases(conversation: Dict[str, Any]) -> List[Dict[str, str]]:
    cid = conversation.get("conversation_id", "")
    turns = conversation.get("messages", [])
    dialogue = "\n".join([f"{m.get('role', '')}: {m.get('text', '')}" for m in turns if m.get("text")])

    prompt = (
        "Voici une conversation entre un client et un agent."
        " Analyse les échanges pour en extraire des cas d’usage sous forme de besoin et solution."
        " Fournis uniquement une liste JSON de dictionnaires avec deux clés : 'besoin' (problème exprimé par le client)"
        " et 'solution' (réponse de l’agent). Réponds en français, sans aucun texte autour du JSON."
        " Exemple :\n"
        "[{\"besoin\": \"Je n’arrive pas à payer avec ma carte.\", \"solution\": \"Essayez une autre carte ou redémarrez l’application.\"}]"
    )

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo-0125",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": f"Conversation:\n{dialogue}"}
            ],
            temperature=0.2
        )
        content = response.choices[0].message.content.strip()
        if not content:
            logger.warning("Empty response from LLM for conversation ID: %s", cid)
            return []
        return json.loads(content)
    except json.JSONDecodeError as json_err:
        logger.error("⚠️ Invalid JSON for conversation %s : %s\nResponse: %s", cid, json_err, content)
        return []
    except Exception as e:
        logger.error("Error extracting use cases for conversation %s: %s", cid, e)
        return []

# Label the conversation with theme, category, and confidence score
def label_conversation_summary(conversation: Dict[str, Any]) -> Dict[str, Any]:
    cid = conversation.get("conversation_id", "")
    texts = [msg.get("text", "") for msg in conversation.get("messages", []) if msg.get("text")]
    full_text = "\n".join(texts).strip()

    # Handle empty conversations
    if not full_text:
        return {"conversation_id": cid, "theme": "unknown", "category": "unknown", "confidence": 0.0}

    # Use cached label if available
    if full_text in label_cache:
        return {"conversation_id": cid, **label_cache[full_text]}

    # System prompt for classification task
    system_prompt = (
        "IMPORTANT : Quelle que soit la langue de la conversation, vous devez d'abord TOUT traduire en français."
        " Puis analysez la conversation traduite et répondez uniquement en français, sans exception."
        "Vous êtes un assistant qui analyse une conversation complète et identifie le thème principal ainsi que la catégorie correspondante."
        " Vous devez généraliser les thèmes et catégories récurrents en un format standardisé."
        " Commencez par traduire la conversation en français si elle est rédigée dans une autre langue (anglais, espagnol, allemand...)."
        " Utilisez une taxonomie standardisée. Répondez UNIQUEMENT en JSON selon la structure suivante :"
        "\n\n"
        "Exemple 1 :\n"
        "{\n  \"theme\": \"Paiement\",\n  \"category\": \"Carte refusée\",\n  \"confidence\": 0.95\n}\n"
        "\n"
        "Exemple 2 :\n"
        "{\n  \"theme\": \"Problème technique\",\n  \"category\": \"La barrière ne s’ouvre pas\",\n  \"confidence\": 0.91\n}\n"
        "\n"
        "Liste possible de thèmes : ['Paiement', 'Connexion', 'Accès', 'Réservation', 'Sortie', 'Compte', 'Information']\n"
        "Liste possible de catégories : ['Carte refusée', 'Mot de passe oublié', 'Code invalide', 'Barrière bloquée', 'Erreur de réservation', 'Compte bloqué', 'Demande de solde']\n"
        "La réponse doit toujours être en français."
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

        # Translate both theme and category to French if needed
        theme = translate_to_french(result.get("theme", "unknown"))
        category = translate_to_french(result.get("category", "unknown"))
        confidence = float(result.get("confidence", 0.0))
        label = {"theme": theme, "category": category, "confidence": confidence}
    except RateLimitError as e:
        logger.error("Quota exceeded. Details: %s", e)
        raise typer.Exit(code=2)
    except Exception as e:
        logger.error("OpenAI error for conversation '%s': %s", full_text, e)
        label = {"theme": "unknown", "category": "unknown", "confidence": 0.0}

    # Store result in cache
    label_cache[full_text] = label
    save_cache()
    return {"conversation_id": cid, **label}

# Command line interface entrypoint to process a batch of conversations
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

    # Process each conversation file
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

                # Handle both list and single dict files
                if isinstance(data, list):
                    for conv in data:
                        label = label_conversation_summary(conv)
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

# Launch CLI app
if __name__ == "__main__":
    app()
