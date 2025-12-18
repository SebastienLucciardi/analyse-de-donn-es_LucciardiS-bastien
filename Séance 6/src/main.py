#coding:utf8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import scipy
import scipy.stats
import math




def ouvrirUnFichier(nom):
    with open(nom, "r") as fichier:
        
        contenu = pd.read_csv(fichier)
    return contenu


def conversionLog(liste):
    log = []
    for element in liste:
        # 
        log.append(math.log(element))
    return log

# trier par ordre décroissant les listes
def ordreDecroissant(liste):
    liste.sort(reverse = True)
    return liste

# Fonction pour obtenir le classement des listes spécifiques aux populations
def ordrePopulation(pop, etat):
    ordrepop = []
    for element in range(0, len(pop)):
        if not np.isnan(pop[element]):
            ordrepop.append([float(pop[element]), etat[element]])
            
    ordrepop = ordreDecroissant(ordrepop)
    
    for element in range(0, len(ordrepop)):
        ordrepop[element] = [element + 1, ordrepop[element][1]]
    
    return ordrepop

# Fonction pour obtenir l'ordre défini entre deux classements 
def classementPays(ordre1, ordre2):
    classement = []
    
    if len(ordre1) <= len(ordre2):
        liste_courte = ordre1
        liste_longue = ordre2
    else:
        liste_courte = ordre2
        liste_longue = ordre1
        
    for rang_val_longue, etat_longue in liste_longue:
        for rang_val_courte, etat_courte in liste_courte:
            if etat_longue == etat_courte:
                if len(ordre1) <= len(ordre2):
                    rang_A = rang_val_courte
                    rang_B = rang_val_longue
                else:
                    rang_A = rang_val_longue
                    rang_B = rang_val_courte
                    
                classement.append([rang_A, rang_B, etat_longue])
                break 
    
    classement.sort() 
    
    return classement

# Fonction pour l'analyse des classements 
def analyseClassementsRangs(classement1, classement2):
    """
    Calcule Spearman's rho et Kendall's tau entre deux classements de rangs
    """
    comparaison = classementPays(classement1, classement2)
    
    rangs1 = [item[0] for item in comparaison]
    rangs2 = [item[1] for item in comparaison]
    
    # scipy.stats.spearmanr et scipy.stats.kendalltau gèrent les ex-aequo 
    spearman, p_spearman = scipy.stats.spearmanr(rangs1, rangs2)
    kendall, p_kendall = scipy.stats.kendalltau(rangs1, rangs2)
    
    return spearman, p_spearman, kendall, p_kendall

# --- Programme Principal ---

chemin_base = "./data/" 

# 1 :

# ouvrir fichier
print("--- Analyse des Îles ---")
try:
    iles_df = ouvrirUnFichier(chemin_base + "island-index.csv")
except FileNotFoundError:
    print(f"Erreur: Le fichier {chemin_base + 'island-index.csv'} n'a pas été trouvé. Vérifiez le chemin.")
    exit()

# 3. Isoler la colonne « Surface (km2) » et ajouter les continents
# Correction du nom de la colonne pour correspondre à l'image fournie : "Surface (km²)"
try:
    surface_iles = iles_df["Surface (km²)"].tolist()
except KeyError:
    # Tente la version sans le caractère spécial si le premier échoue
    try:
        surface_iles = iles_df["Surface (km2)"].tolist()
    except KeyError:
        print("Erreur: La colonne 'Surface (km²)' ou 'Surface (km2)' est manquante. Vérifiez le nom dans le fichier.")
        exit()

# Nettoyage et typage: forcer le typage en float()
surface_final = []
for item in surface_iles:
    try:
        surface_final.append(float(item))
    except ValueError:
        pass # Ignore les valeurs non numériques

# Ajout des masses continentales (en km2, sans unité)
masses_continentales = [
    85545323.0,  
    37856841.0,  
    7768030.0,   
    7605049.0    
]
surface_final.extend(masses_continentales)

# 4. Ordonner la liste obtenue 
surface_ordonnee = ordreDecroissant(surface_final)
rangs = list(range(1, len(surface_ordonnee) + 1))


# 5. Visualiser la loi rang-taille (axes normaux)
plt.figure(figsize=(10, 6))
plt.scatter(rangs, surface_ordonnee)
plt.title("Loi Rang-Taille (Axes Normaux)")
plt.xlabel("Rang")
plt.ylabel("Surface (km²)")
plt.grid(True)
plt.show()


# 6. Convertir les axes en logarithme (pour la lisibilité)
surface_log = conversionLog(surface_ordonnee)
rangs_log = conversionLog(rangs)

# Visualisation en coordonnées log-log
plt.figure(figsize=(10, 6))
plt.scatter(rangs_log, surface_log)
plt.title("Loi Rang-Taille (Axes Log-Log)")
plt.xlabel("Log(Rang)")
plt.ylabel("Log(Surface)")
plt.grid(True)
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(rangs_log, surface_log)
plt.plot(rangs_log, intercept + slope*np.array(rangs_log), 'r', label=f'Régression linéaire (pente={slope:.2f})')
plt.legend()
plt.show()


# 7. Est-il possible de faire un test sur les rangs ?
print("\n# 7. Test sur les rangs")
"""
Réponse : 
Oui, il est possible de faire un test sur les rangs. On peut utiliser l'analyse de régression linéaire sur le graphe Log-Log 
pour vérifier l'adéquation à la loi de Zipf (test de la pente $\alpha$ par rapport à la théorie). 
De plus, des tests de corrélation (comme Spearman) peuvent être utilisés pour mesurer la force de l'ordre.
"""

# --- PARTIE 2 : POPULATIONS MONDIALES ET CORRÉLATION DES RANGS ---

# 8. & 9. Ouvrir le fichier Etats-du-monde
print("\n--- Analyse des Populations Mondiales ---")
try:
    monde_df = ouvrirUnFichier(chemin_base + "Le-Monde-HS-Etats-du-monde-2007-2025.csv")
except FileNotFoundError:
    print(f"Erreur: Le fichier {chemin_base + 'Le-Monde-HS-Etats-du-monde-2007-2025.csv'} n'a pas été trouvé. Vérifiez le chemin.")
    exit()

# 10. Isoler les colonnes
etats = monde_df["État"].tolist()
pop_2007 = monde_df["Pop 2007"].tolist()
pop_2025 = monde_df["Pop 2025"].tolist()
densite_2007 = monde_df["Densité 2007"].tolist()
densite_2025 = monde_df["Densité 2025"].tolist()

# 11. Ordonner de manière décroissante les listes
classement_pop_2007 = ordrePopulation(pop_2007, etats)
classement_pop_2025 = ordrePopulation(pop_2025, etats)
classement_densite_2007 = ordrePopulation(densite_2007, etats)
classement_densite_2025 = ordrePopulation(densite_2025, etats)

print(f"Nombre d'États classés en Pop 2007: {len(classement_pop_2007)}")
print(f"Nombre d'États classés en Pop 2025: {len(classement_pop_2025)}")

# 12. Préparer la comparaison des listes sur la population et la densité
comparaison_pop = classementPays(classement_pop_2007, classement_pop_2025)
comparaison_densite = classementPays(classement_densite_2007, classement_densite_2025)

# 13.
# Population
rangs_pop_2007 = [item[0] for item in comparaison_pop]
rangs_pop_2025 = [item[1] for item in comparaison_pop]
# Densité
rangs_densite_2007 = [item[0] for item in comparaison_densite]
rangs_densite_2025 = [item[1] for item in comparaison_densite]

# 14. 
# --- Population (2007 vs 2025) ---
spearman_pop, p_spearman_pop = scipy.stats.spearmanr(rangs_pop_2007, rangs_pop_2025)
kendall_pop, p_kendall_pop = scipy.stats.kendalltau(rangs_pop_2007, rangs_pop_2025)

print("\nCorrélation des Rangs (Population 2007 vs 2025):")
print(f"  - Spearman (rho): {spearman_pop:.4f} (p-value: {p_spearman_pop:.4f})")
print(f"  - Kendall (tau): {kendall_pop:.4f} (p-value: {p_kendall_pop:.4f})")

# --- Densité (2007 vs 2025) ---
spearman_densite, p_spearman_densite = scipy.stats.spearmanr(rangs_densite_2007, rangs_densite_2025)
kendall_densite, p_kendall_densite = scipy.stats.kendalltau(rangs_densite_2007, rangs_densite_2025)

print("\nCorrélation des Rangs (Densité 2007 vs 2025):")
print(f"  - Spearman (rho): {spearman_densite:.4f} (p-value: {p_spearman_densite:.4f})")
print(f"  - Kendall (tau): {kendall_densite:.4f} (p-value: {p_kendall_densite:.4f})")


# --- PARTIE BONUS 2.3 ---

print("\n--- Partie Bonus : Fonctions d'Analyse de Rangs ---")

## Bonus 

# Correction des noms de colonnes basés sur l'image : "Surface (km²)", "Trait de côte (km)", "Toponyme"
try:
    surface_iles_original = iles_df["Surface (km²)"].tolist() 
    trait_cote_iles = iles_df["Trait de côte (km)"].tolist()
    noms_iles = iles_df["Toponyme"].tolist() 
except KeyError as e:
    print(f"Erreur: La colonne n'a pas été trouvée pour la partie Bonus. Erreur: {e}")
    

try:
    classement_surface = ordrePopulation(surface_iles_original, noms_iles)
    classement_trait_cote = ordrePopulation(trait_cote_iles, noms_iles)

    comparaison_iles = classementPays(classement_surface, classement_trait_cote)

    rangs_surface = [item[0] for item in comparaison_iles]
    rangs_trait_cote = [item[1] for item in comparaison_iles]

    spearman_iles, p_spearman_iles = scipy.stats.spearmanr(rangs_surface, rangs_trait_cote)
    kendall_iles, p_kendall_iles = scipy.stats.kendalltau(rangs_surface, rangs_trait_cote)

    print("\nCorrélation des Rangs (Îles: Surface vs Trait de Côte):")
    print(f"  - Spearman (rho): {spearman_iles:.4f} (p-value: {p_spearman_iles:.4f})")
    print(f"  - Kendall (tau): {kendall_iles:.4f} (p-value: {p_kendall_iles:.4f})")
except NameError:
    # Cette erreur est générée si la partie try/except précédente n'a pas réussi à définir les listes
    print("\nImpossible d'exécuter l'analyse de corrélation des îles (Surface vs Trait de Côte) car les données n'ont pas été chargées correctement.")


## Bonus : 

# 1. Test de la fonction factorisée (voir page 8)
spearman_pop_f, p_spearman_pop_f, kendall_pop_f, p_kendall_pop_f = analyseClassementsRangs(classement_pop_2007, classement_pop_2025)
print("\nTest de la fonction factorisée (Pop 2007 vs 2025):")
print(f"  - Spearman (rho): {spearman_pop_f:.4f} | Kendall (tau): {kendall_pop_f:.4f}")


# 2. Analyse de la concordance des rangs 

annees = list(range(2007, 2026))
resultats_annuels_pop = []

ref_classement = classement_pop_2007 

for annee in annees:
    col_pop = f"Pop {annee}"
    
    try:
        pop_annee = monde_df[col_pop].tolist()
    except KeyError:
        continue 
        
    classement_annee = ordrePopulation(pop_annee, etats)
    
    if annee == 2007:
        spearman, kendall = 1.0, 1.0
    else:
        #fonction factorisée
        spearman, _, kendall, _ = analyseClassementsRangs(ref_classement, classement_annee)
        
    resultats_annuels_pop.append({
        'Année': annee,
        'Spearman_rho': spearman,
        'Kendall_tau': kendall
    })

df_resultats_annuels = pd.DataFrame(resultats_annuels_pop)
print("\nConcordance des Rangs de Population (vs 2007):")
print(df_resultats_annuels.to_string(index=False))

# tendance
plt.figure(figsize=(12, 6))
plt.plot(df_resultats_annuels['Année'], df_resultats_annuels['Spearman_rho'], marker='o', label="Spearman's rho")
plt.plot(df_resultats_annuels['Année'], df_resultats_annuels['Kendall_tau'], marker='x', label="Kendall's tau")
plt.title("Évolution de la Concordance des Rangs de Population (Référence 2007)")
plt.xlabel("Année")
plt.ylabel("Coefficient de Corrélation")
plt.ylim(min(df_resultats_annuels['Spearman_rho']) * 0.99, 1.005) 
plt.legend()
plt.grid(True)
plt.show()