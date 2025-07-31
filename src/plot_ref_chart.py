import json
import typer
import matplotlib.pyplot as plt
from collections import Counter

app = typer.Typer()

@app.command()
def plot_chart(
    ref_file: str = typer.Option(..., "--ref-file", help="Path to ref.json"),
    output_file: str = typer.Option("top_categories_chart.png", "--output-file", help="Output image path")
):
    with open(ref_file, "r", encoding="utf-8") as f:
        ref = json.load(f)

    all_categories = []
    for theme in ref["themes"]:
        for cat in theme["categories"]:
            all_categories.append((theme["theme"], cat["category"], cat["frequency"]))

    # Trier par fréquence globale
    sorted_cats = sorted(all_categories, key=lambda x: x[2], reverse=True)[:15]

    labels = [f"{theme} > {cat}" for theme, cat, _ in sorted_cats]
    freqs = [freq for _, _, freq in sorted_cats]

    plt.figure(figsize=(12, 6))
    plt.barh(labels[::-1], freqs[::-1])
    plt.title("Top 15 catégories (thème > catégorie)")
    plt.xlabel("Fréquence")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    print(f"✅ Graph saved to {output_file}")

if __name__ == "__main__":
    app()
