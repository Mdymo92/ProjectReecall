import json
from collections import defaultdict
from typing import Dict, List
import typer
from tqdm import tqdm
from rapidfuzz import fuzz

app = typer.Typer()

FUZZY_THRESHOLD = 85  # Similarity score threshold for merging

def find_closest_category(name: str, existing: List[str]) -> str:
    best_match = None
    highest_score = 0
    for candidate in existing:
        score = fuzz.partial_ratio(name.lower(), candidate.lower())
        if score > highest_score:
            best_match = candidate
            highest_score = score
    return best_match if highest_score >= FUZZY_THRESHOLD else name

@app.command()
def build_ref(
    labels_path: str = typer.Argument(..., help="Path to labels_output.jsonl"),
    output_path: str = typer.Argument("ref.json", help="Output file path for the reference JSON")
):
    theme_data: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
        "frequency": 0,
        "examples": []
    }))
    category_aliases: Dict[str, str] = {}  # Raw -> Canonical mapping per theme

    with open(labels_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Processing conversations"):
            obj = json.loads(line.strip())
            theme = obj.get("theme", "unknown")
            raw_category = obj.get("categorie", "unknown")
            use_cases = obj.get("use_cases", [])

            # Find canonical name for the category within the theme
            if theme not in category_aliases:
                category_aliases[theme] = {}
            aliases = category_aliases[theme]

            if raw_category in aliases:
                canonical = aliases[raw_category]
            else:
                known_categories = list(theme_data[theme].keys())
                canonical = find_closest_category(raw_category, known_categories)
                aliases[raw_category] = canonical

            # Register frequency and examples
            theme_data[theme][canonical]["frequency"] += 1
            for case in use_cases:
                if "besoin" in case and len(theme_data[theme][canonical]["examples"]) < 2:
                    theme_data[theme][canonical]["examples"].append(case["besoin"])

    # Format output
    ref_output: List[Dict] = []
    for theme, categories in sorted(
        theme_data.items(),
        key=lambda t: sum(c["frequency"] for c in t[1].values()),
        reverse=True
    ):
        ref_output.append({
            "theme_id": len(ref_output),
            "theme": theme,
            "frequency": sum(c["frequency"] for c in categories.values()),
            "categories": [
                {
                    "category_id": idx,
                    "category": cat,
                    "frequency": data["frequency"],
                    "examples": data["examples"]
                }
                for idx, (cat, data) in enumerate(
                    sorted(categories.items(), key=lambda item: item[1]["frequency"], reverse=True)
                )
            ]
        })

    with open(output_path, "w", encoding="utf-8") as fout:
        json.dump({"themes": ref_output}, fout, ensure_ascii=False, indent=2)

    print(f"✅ Reference file with fuzzy category grouping saved to: {output_path}")

if __name__ == "__main__":
    app()
