

import pandas as pd
import matplotlib.pyplot as plt

# Source des données : https://www.data.gouv.fr/datasets/election-presidentielle-des-10-et-24-avril-2022-resultats-definitifs-du-1er-tour/
with open("./data/resultats-elections-presidentielles-2022-1er-tour.csv","r") as fichier:
    contenu = pd.read_csv(fichier)

# Question 1
# ...
import pandas as pd

# 1️Lecture du fichier CSV avec Pandas
chemin_fichier = "data/resultats-elections-presidentielles-2022-1er-tour.csv"

with open(chemin_fichier, "r") as fichier:
    contenu = pd.read_csv(fichier, sep=',', quotechar='"')

# 2️Afficher le contenu
print("=== Aperçu du contenu du CSV ===")
print(contenu)

# 3️ Nombre de lignes et colonnes
print("\n=== Dimensions du tableau ===")
print(f"Lignes : {len(contenu)}")
print(f"Colonnes : {len(contenu.columns)}")

# 4️Types de variables (nature statistique)
print("\n=== Types des colonnes ===")
types_colonnes = {col: contenu[col].dtype for col in contenu.columns}
for col, type_ in types_colonnes.items():
    print(f"{col} : {type_}")

# 5️Noms des colonnes
print("\n=== Noms des colonnes ===")
print(contenu.head(0))  # Affiche uniquement la ligne d’en-tête

# 6️Sélection du nombre d'inscrits
print("\n=== Colonne 'Inscrits' ===")
if 'Inscrits' in contenu.columns:
    print(contenu['Inscrits'])
else:
    print("⚠ La colonne 'Inscrits' n'existe pas (vérifie le nom exact avec head())")

# 7️Calcul des sommes (uniquement pour les données quantitatives)
print("\n=== Sommes des colonnes numériques ===")
for col in contenu.columns:
    if pd.api.types.is_numeric_dtype(contenu[col]):
        print(f"{col} : {contenu[col].sum()}")
        import os
import matplotlib.pyplot as plt

# === Étape 11 : Diagrammes en barres (Inscrits / Votants) ===

# Créer un dossier "images/barres" s'il n'existe pas déjà
os.makedirs("images/barres", exist_ok=True)

for i, ligne in contenu.iterrows():
    departement = ligne["Libellé du département"]
    inscrits = ligne["Inscrits"]
    votants = ligne["Votants"]

    plt.figure(figsize=(5, 3))
    plt.bar(["Inscrits", "Votants"], [inscrits, votants])
    plt.title(f"{departement} - Inscrits vs Votants")
    plt.ylabel("Nombre de personnes")

    # Sauvegarde de l'image en .png
    plt.tight_layout()
    plt.savefig(f"images/barres/{departement.replace('/', '_')}.png")
    plt.close()

print("Diagrammes en barres créés dans le dossier images/barres/")


# === Étape 12 : Diagrammes circulaires (Blancs / Nuls / Exprimés / Abstentions) ===

os.makedirs("images/camemberts", exist_ok=True)

for i, ligne in contenu.iterrows():
    departement = ligne["Libellé du département"]
    blancs = ligne["Blancs"]
    nuls = ligne["Nuls"]
    exprimes = ligne["Exprimés"]
    abstentions = ligne["Abstentions"]

    valeurs = [blancs, nuls, exprimes, abstentions]
    labels = ["Blancs", "Nuls", "Exprimés", "Abstentions"]

    plt.figure(figsize=(4, 4))
    plt.pie(valeurs, labels=labels, autopct="%1.1f%%", startangle=90)
    plt.title(f"{departement} - Répartition des votes")

    plt.tight_layout()
    plt.savefig(f"images/camemberts/{departement.replace('/', '_')}.png")
    plt.close()

print(" Diagrammes circulaires créés dans le dossier images/camemberts/")


# === Étape 13 : Histogramme de la distribution des inscrits ===

os.makedirs("images", exist_ok=True)

plt.figure(figsize=(8, 5))
plt.hist(contenu["Inscrits"], bins=20, density=True, edgecolor="black")
plt.title("Distribution du nombre d'inscrits")
plt.xlabel("Nombre d'inscrits")
plt.ylabel("Fréquence")
plt.tight_layout()
plt.savefig("images/histogramme_inscrits.png")
plt.close()

print(" Histogramme des inscrits créé dans images/histogramme_inscrits.png")
