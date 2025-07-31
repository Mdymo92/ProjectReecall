import json
from collections import defaultdict
from typing import Dict, List
import typer
from tqdm import tqdm

app = typer.Typer()

@app.command()
def build_ref(
    labels_path: str = typer.Argument(..., help="Path to labels_output.json"),
    output_path: str = typer.Argument("ref.json", help="Output file path for the reference JSON")
):
    # Structure: theme -> category -> data (frequency + examples)
    theme_data: Dict[str, Dict[str, Dict]] = defaultdict(lambda: defaultdict(lambda: {
        "frequency": 0,
        "examples": []
    }))

    with open(labels_path, "r", encoding="utf-8") as fin:
        for line in tqdm(fin, desc="Processing conversations"):
            obj = json.loads(line.strip())
            theme = obj.get("theme", "unknown")
            category = obj.get("categorie", "unknown")
            use_cases = obj.get("use_cases", [])

            # Count frequency and collect up to 2 examples per category
            theme_data[theme][category]["frequency"] += 1
            for case in use_cases:
                if "besoin" in case and len(theme_data[theme][category]["examples"]) < 2:
                    theme_data[theme][category]["examples"].append(case["besoin"])

    # Convert to structured list ordered by frequency
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

    print(f"✅ Reference file generated: {output_path}")

if __name__ == "__main__":
    app()
