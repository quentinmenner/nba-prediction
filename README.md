# Prédiction sur les résultats de la NBA
## Projet de Python pour le Data Scientist - Janvier 2022 - Nicolas Drettakis et Quentin Menner

Le rapport est disponible dans le fichier rapport.md. Le dossier images contient les tableaux utilisés dans le rapport.

### Webscraping
Les programmes de webscraping menant à la création des deux dataframes (`stats-equipes-par-mois.csv` et `stats-matchs.csv`) contenant les données sont disponibles dans `webscraping-equipes.ipynb` et `webscraping-match.ipynb`. Le fichier `chromedriver.exe` est nécessaire pour utiliser sélenium.

### Statistiques Descriptives
L'intégralité du code concernant les statistiques descriptives est contenu dans le fichier Statistiques descriptives.ipynb. Il utilise le tableau tableaupython.csv qui contient les équipes premières de la saison régulière de 2005 à 2019. 

### Modèle
Les fichiers nécessaires pour le Random Forest sont créés par le programme `pre_process.ipynb`. Le modèle Random Forest est exécuté dans le fichier `modelisation.ipynb` et utilise les dataframes `df_test_1.csv`, `df_test_2.csv`, `df_train_1.csv` et `df_train_2.csv` pour séparer les données en une partie pour entraîner le modèle et une autre pour le tester.
