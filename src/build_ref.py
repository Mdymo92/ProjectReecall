import json
from collections import defaultdict, Counter
from typing import Dict, List
import typer
from tqdm import tqdm
import openai
import os
from dotenv import load_dotenv

app = typer.Typer()

# Load .env file if it exists
load_dotenv()
openai.api_key = os.getenv("OPENAI_API_KEY")

if not openai.api_key:
    raise EnvironmentError("❌ OPENAI_API_KEY is not set. Please set it in your environment or .env file.")

@app.command()
def regroup_ref_llm(
    labels_path: str = typer.Argument(..., help="Path to labels_output.jsonl"),
    output_path: str = typer.Argument("ref_llm.json", help="Output path for regrouped reference")
):
    cleaned_lines = []
    output_clean_path = "labels_output_clean.json"
    with open(labels_path, "r", encoding="utf-8") as fin, open(output_clean_path, "w", encoding="utf-8") as fout:
        for i, line in enumerate(fin, 1):
            try:
                json_obj = json.loads(line.strip())
                fout.write(json.dumps(json_obj, ensure_ascii=False) + "\n")
                cleaned_lines.append(json_obj)
            except json.JSONDecodeError as e:
                print(f"❌ Ligne {i} invalide: {e}")

    label_pairs = []
    counts = Counter()
    examples: Dict[str, Dict[str, List[str]]] = defaultdict(lambda: defaultdict(list))

    for conv in cleaned_lines:
        theme = conv.get("theme", "inconnu")
        category = conv.get("categorie", "inconnu")
        label_pairs.append({"theme": theme, "categorie": category})
        counts[(theme, category)] += 1
        for uc in conv.get("use_cases", []):
            if "besoin" in uc and len(examples[theme][category]) < 2:
                examples[theme][category].append(uc["besoin"])

    prompt = (
    "Voici une liste de paires thème / catégorie extraites de conversations clients avec leurs fréquences.\n"
    "Regroupe les catégories similaires entre elles et associe-les à des thèmes cohérents.\n"
    "Fournis un JSON structuré avec les clés suivantes :\n"
    "{\n"
    "  'themes': [\n"
    "    {\n"
    "      'theme_id': int (identifiant unique),\n"
    "      'theme': str (nom du thème en français),\n"
    "      'frequency': int (somme des fréquences des catégories de ce thème),\n"
    "      'categories': [\n"
    "        {\n"
    "          'category_id': int (identifiant unique de la catégorie dans le thème),\n"
    "          'category': str (nom de la catégorie en français),\n"
    "          'frequency': int (nombre d'occurrences),\n"
    "          'examples': list[str] (exemples en français)\n"
    "        },\n"
    "        ...\n"
    "      ]\n"
    "    },\n"
    "    ...\n"
    "  ]\n"
    "}\n"
    "Utilise impérativement ces noms de clés en anglais pour que le fichier soit lisible par une machine, "
    "mais rédige tous les contenus (noms de thèmes, catégories, exemples) en français.\n"
    "Réponds uniquement avec ce JSON.")


    content_to_send = json.dumps([
        {"theme": k[0], "categorie": k[1], "frequency": v, "examples": examples[k[0]][k[1]]}
        for k, v in counts.items()
    ], ensure_ascii=False)

    try:
        response = openai.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": content_to_send}
            ],
            temperature=0.3
        )
        output = response.choices[0].message.content.strip()
        parsed = json.loads(output)
        with open(output_path, "w", encoding="utf-8") as fout:
            json.dump(parsed, fout, ensure_ascii=False, indent=2)
        print(f"✅ LLM-based reference saved to: {output_path}")
    except Exception as e:
        print(f"❌ Error calling OpenAI: {e}")

if __name__ == "__main__":
    app()
