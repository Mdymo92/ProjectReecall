### 🔐 Recommandation d'utilisation d'un service d'IA (OpenAI) dans le respect du RGPD

Dans le cadre de ce projet, nous recommandons l'utilisation de l'API **OpenAI (GPT-3.5 / GPT-4)** pour les tâches d’annotation sémantique, de classification thématique et de génération de cas d’usage. Cette solution a été retenue pour ses performances, sa flexibilité et sa capacité à traiter plusieurs langues.

#### ✅ Avantages du service

* Excellente compréhension contextuelle des conversations client
* Réduction significative du temps d’annotation
* Facilité d’intégration via API
* Traduction automatique multilingue

#### 🛡️ Respect des contraintes réglementaires (RGPD)

Pour garantir la conformité avec le **Règlement Général sur la Protection des Données (RGPD)**, les recommandations suivantes doivent impérativement être respectées :

* **Anonymisation stricte** : Toutes les conversations envoyées à l’API doivent être préalablement **anonymisées** (suppression des noms, adresses, numéros, emails…).
* **Pas de données sensibles** : Aucune **donnée personnelle ou confidentielle** ne doit transiter via l’API.
* **Utilisation encadrée** : L’usage d’OpenAI doit être limité à des tâches **non-déterminantes** pour l’utilisateur final (pré-analyse, prototype, exploration).
* **Stockage local** : Ne stocker aucune réponse contenant des informations critiques sans anonymisation préalable.
* **Clé API protégée** : La clé `OPENAI_API_KEY` doit être conservée dans un fichier `.env` non versionné (`.gitignore`) et non diffusé.

#### 🔒 Bonnes pratiques recommandées

* ✔️ Ajouter une couche d’audit pour tracer les appels à l’API
* ✔️ Mettre en place une rotation périodique de la clé API
* ✔️ Ne jamais exposer l’API dans des environnements accessibles au public (frontend, navigateur)


