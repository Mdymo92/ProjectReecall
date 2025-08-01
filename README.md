# Test technique Reecall (Ilyes Diny)
![Python Badge](https://img.shields.io/badge/Python-3.13-%233776AB?logo=python&logoColor=%233776AB)



## Executer le projet 
### 1) Cloner le repo et créer l’environnement virtuel
- `git clone <https://github.com/Mdymo92/TestReecall.git>` 
- `cd TestReecall`
- `python3 -m venv venv`

### 2) Activer l’environnement
`venv\Scripts\activate.bat` (# Sur Windows (CMD))

### 3) 🧪 Installer les dépendances
- `pip install -r requirements.txt`
- [make](https://www.gnu.org/software/make/). L’utilisation de `make` ici est optionnelle car toutes les opérations sur l’environnement virtuel (`venv`) peuvent être effectuées normalement ; il a seulement été utilisé pour offrir une meilleure expérience développeur.
- [python3.13](https://docs.python.org/3/whatsnew/3.13.html)
- [python3.13-venv](https://docs.python.org/3/tutorial/venv.html)

### 🔐 Variables d’environnement
Ajoutez une clé OpenAI (non Free Tier) dans un fichier `.env` à créer dans le dossier `src` : 

`OPENAI_API_KEY=...`


## ⚙️ Pipeline étape par étape

### 1. 🔄 Ingestion des données

Structure les données anonymisées au format standard :

`python src/ingest.py --input-dir src/ANONYMIZATION/ --output-dir src/interm/`

### 2. 🧼 Prétraitement (nettoyage, unicité, formatage)

Nettoie les conversations pour les rendre exploitables :

`python src/preprocess.py --input-dir src/interm/ --output-dir src/clean/ --batch-size 10`

### 3. 🧠 Étiquetage avec LLM (GPT)

Attribue à chaque conversation un thème, une catégorie, une confiance, et des cas d’usage :

`python src/semantic.py src/clean/ src/labels_output.json --pattern "*.jsonc"`

### 4. 🧱 Construction du référentiel thématique

Regroupe les thèmes et catégories par similarité, et génère la structure du ref.json avec fréquence et exemples :

`python src/build_ref.py src/labels_output.json src/ref.json`

### 5. 📊 Visualisation des catégories dominantes

Génère un graphique à barres des thèmes et catégories les plus fréquents :

`python src/plot_ref_chart.py --ref-file src/ref.json --output-file src/top_categories_chart.png`


### 📁 Fichiers générés

* `src/interm/`: Donnees brutes reorganisees par conversation depuis les fichiers d'entree  
* `src/clean/` : Conversations nettoyees et au format standard JSONC  
* `labels_output.json` : Conversations annotées avec les thèmes, catégories et cas d’usage
* `ref.json` : Taxonomie finale des thèmes et catégories avec fréquences et exemples
* `top_categories_chart.png` : Résumé visuel des catégories les plus fréquentes


### ✅ Statut
Cette pipeline est complète et modulaire. Chaque étape peut être exécutée indépendamment.



##
<a href="https://gitmoji.dev">
  <img
    src="https://img.shields.io/badge/gitmoji-%20😜%20😍-FFDD67.svg?style=flat-square"
    alt="Gitmoji"
  />
</a>
