#coding:utf8

import pandas as pd
import numpy as np
import scipy.stats

FILE_PATH_ECHANTILLONS = "/Users/hsen/Desktop/STT/data/Echantillonnage-100-Echantillons.csv"
FILE_PATH_LOI1 = "/Users/hsen/Desktop/STT/data/Loi-normale-Test-1.csv"
FILE_PATH_LOI2 = "/Users/hsen/Desktop/STT/data/Loi-normale-Test-2.csv"

def ouvrirUnFichier(chemin_fichier):
    try:
        df = pd.read_csv(chemin_fichier, sep=';')
        
        if df.shape[1] == 1:
            try:
                df = pd.read_csv(chemin_fichier, sep=',')
            except:
                df = pd.read_csv(chemin_fichier, sep=r'\s+', engine='python', skipinitialspace=True)
        
        df = df.dropna(axis=1, how='all')

        if 'Echantillonnage' in chemin_fichier and df.shape[1] == 3:
            df.columns = ['Pour', 'Contre', 'Sans_opinion']
        
        if df.shape[1] != 3 and 'Echantillonnage' in chemin_fichier:
            print(f"ATTENTION: Le fichier {chemin_fichier} n'a pu être lu qu'avec {df.shape[1]} colonnes. Le séparateur est peut-être incorrect.")
        
        return df
    
    except FileNotFoundError:
        print(f"ERREUR: Le fichier {chemin_fichier} n'a pas été trouvé. Veuillez vérifier le chemin.")
        return None
    except Exception as e:
        print(f"ERREUR lors de l'ouverture du fichier {chemin_fichier}: {e}")
        return None

df_echantillons = ouvrirUnFichier(FILE_PATH_ECHANTILLONS)

print("="*60)
print("ÉTAPE 1 : THÉORIE DE L’ÉCHANTILLONNAGE")
print("="*60)

if df_echantillons is not None:
    N_mere = 2185
    pour_mere = 852
    contre_mere = 911
    sans_opinion_mere = 422
    
    moyennes_echantillons = df_echantillons.mean().round(0).astype(int)
    print("\n[1.1] Moyennes des 100 échantillons (arrondies) :")
    print(moyennes_echantillons)
    
    somme_moyennes = moyennes_echantillons.sum()
    frequences_echantillons = (moyennes_echantillons / somme_moyennes).round(2)
    print("\n[1.2] Fréquences des moyennes des échantillons (arrondies à 2 décimales) :")
    print(frequences_echantillons)
    
    effectifs_mere = pd.Series([pour_mere, contre_mere, sans_opinion_mere],
                                 index=['Pour', 'Contre', 'Sans_opinion'])
    frequences_mere = (effectifs_mere / N_mere).round(2)
    print("\n[1.3] Fréquences de la population mère (arrondies à 2 décimales) :")
    print(frequences_mere)

    z_C = 1.96
    n_moyen = somme_moyennes
    
    print(f"\n[1.4] Intervalle de fluctuation (IF) à 95% (n moyen={n_moyen}) :")
    
    intervalle_fluctuation = {}
    for opinion, p in frequences_mere.items():
        ecart_type_echantillonnage = np.sqrt(p * (1 - p) / n_moyen)
        marge_erreur = z_C * ecart_type_echantillonnage
        
        borne_inf = p - marge_erreur
        borne_sup = p + marge_erreur
        
        intervalle_fluctuation[opinion] = (f"{borne_inf:.4f}", f"{borne_sup:.4f}")
        print(f"IF '{opinion}' (p={p:.2f}): [{borne_inf:.4f} ; {borne_sup:.4f}]")


print("\n"+"="*60)
print("ÉTAPE 2 : THÉORIE DE L’ESTIMATION")
print("="*60)

if df_echantillons is not None:
    premier_echantillon_pandas = df_echantillons.iloc[0]
    
    premier_echantillon_list = premier_echantillon_pandas.tolist()
    print(f"\n[2.1] Premier échantillon (liste) : {premier_echantillon_list}")

    n_echantillon_1 = premier_echantillon_pandas.sum()
    print(f"Taille de l'échantillon (n) : {n_echantillon_1}")

    frequences_echantillon_1 = (premier_echantillon_pandas / n_echantillon_1)
    print("\n[2.3] Fréquences du premier échantillon (p_hat) :")
    print(frequences_echantillon_1.round(4))

    z_C = 1.96
    
    print(f"\n[2.4] Intervalle de confiance (IC) à 95% (n={n_echantillon_1}) :")
    
    for opinion, p_hat in frequences_echantillon_1.items():
        ecart_type_estimation = np.sqrt(p_hat * (1 - p_hat) / n_echantillon_1)
        marge_erreur = z_C * ecart_type_estimation
        
        borne_inf = p_hat - marge_erreur
        borne_sup = p_hat + marge_erreur
        
        print(f"IC '{opinion}' (p_hat={p_hat:.4f}): [{borne_inf:.4f} ; {borne_sup:.4f}]")


print("\n"+"="*60)
print("ÉTAPE 3 : THÉORIE DE LA DÉCISION (Test de Shapiro-Wilks)")
print("="*60)

df_loi1 = ouvrirUnFichier(FILE_PATH_LOI1)
df_loi2 = ouvrirUnFichier(FILE_PATH_LOI2)
alpha = 0.05

def realiser_shapiro_wilks(df, nom_fichier, alpha):
    if df is not None:
        data = df.iloc[:, 0].dropna()
        if len(data) < 3 or len(data) > 5000:
            print(f"ATTENTION: Le test de Shapiro-Wilks est moins adapté pour {nom_fichier} (taille={len(data)}).")
        
        stat_sw, p_value = scipy.stats.shapiro(data)
        
        print(f"\nRésultats pour {nom_fichier} (Taille: {len(data)})")
        print(f"Statistique W : {stat_sw:.4f}")
        print(f"P-value : {p_value:.4f}")
        
        if p_value > alpha:
            conclusion = f"P-value ({p_value:.4f}) > alpha ({alpha}). Ne rejette pas H0."
            resultat = "La distribution est considérée comme NORMALE."
        else:
            conclusion = f"P-value ({p_value:.4f}) <= alpha ({alpha}). Rejette H0."
            resultat = "La distribution n'est PAS considérée comme normale."
        
        print(f"Conclusion : {conclusion}")
        print(f"Résultat : {resultat}")
        return resultat
    return "Non exécuté"

res_loi1 = realiser_shapiro_wilks(df_loi1, "Loi-normale-Test-1.csv", alpha)
res_loi2 = realiser_shapiro_wilks(df_loi2, "Loi-normale-Test-2.csv", alpha)

print("\n"+"="*60)
print("FIN DES CALCULS STATISTIQUES.")
print("="*60)

print("\n"+"="*60)
print("ANALYSE BONUS : LOIS NON NORMALES")
print("="*60)

def analyser_distribution(df, nom):
    if df is not None:
        data = df.iloc[:, 0].dropna()
        
        statistiques = {
            "Taille (n)": len(data),
            "Minimum": data.min(),
            "Maximum": data.max(),
            "Étendue (Max - Min)": data.max() - data.min(),
            "Moyenne": data.mean(),
            "Médiane": data.median(),
            "Écart-type": data.std(),
        }
        
        print(f"\nStatistiques descriptives pour {nom} :")
        for cle, valeur in statistiques.items():
            print(f"- {cle:<20}: {valeur:.4f}")
            
        print("\n=> Caractérisation (pour le Bonus) :")
        
        if abs(statistiques["Moyenne"] - statistiques["Médiane"]) > 0.1 * statistiques["Écart-type"]:
            print("    La moyenne et la médiane sont éloignées. La distribution est fortement asymétrique.")
            print("    L'hypothèse d'une loi Exponentielle ou d'une autre loi asymétrique est forte.")
        
        elif abs(statistiques["Moyenne"] - (statistiques["Minimum"] + statistiques["Maximum"]) / 2) < 0.05 * statistiques["Écart-type"]:
            print("    La moyenne est très proche du centre de l'intervalle [Min, Max].")
            print("    L'hypothèse d'une Loi Uniforme est forte, car l'écart-type est faible par rapport à l'étendue.")
        
        else:
            print("    Les statistiques ne permettent pas de trancher facilement sans visualisation (histogramme).")


analyser_distribution(ouvrirUnFichier(FILE_PATH_LOI1), "Loi-normale-Test-1.csv")
analyser_distribution(ouvrirUnFichier(FILE_PATH_LOI2), "Loi-normale-Test-2.csv")