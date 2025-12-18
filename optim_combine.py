import pickle
import re
from pathlib import Path
import calendar

###############################################################################
# CONFIG
###############################################################################

# Un ou plusieurs dossiers contenant les gridsearch_results_*.pkl
# Tu peux mettre un seul dossier si tu veux.
result_dirs = [
    Path("C:/Users/Vincent/Downloads/train/optim/"),         # RNN
    Path("C:/Users/Vincent/Downloads/train/optim/lin/"),  # linéaire
]

# Nom du fichier de sortie fusionné
output_file = Path("ALL_gridsearch_results_v2.pkl")

# ---- mois à EXCLURE (laisse vide pour ne rien exclure) ----
# exemples acceptés: 8, '08', 'Aug', 'August', 'août', 'aout', etc.
exclude_months_input = []   # [] => ne retire aucune date
# -----------------------------------------------------------

###############################################################################
# UTILITAIRES MOIS
###############################################################################
MONTH_LOOKUP = {name.lower(): i for i, name in enumerate(calendar.month_name) if name}
MONTH_LOOKUP.update({name.lower(): i for i, name in enumerate(calendar.month_abbr) if name})
MONTH_LOOKUP.update({'août': 8, 'aout': 8})

def to_month_int(x):
    if isinstance(x, int):
        if 1 <= x <= 12:
            return x
        raise ValueError(f"Invalid month int: {x}")
    s = str(x).strip().lower()
    if s.isdigit():
        m = int(s)
        if 1 <= m <= 12:
            return m
        raise ValueError(f"Invalid month number: {x}")
    if s in MONTH_LOOKUP:
        return MONTH_LOOKUP[s]
    raise ValueError(f"Unknown month label: {x}")

exclude_months = {to_month_int(m) for m in exclude_months_input}

def extract_month_from_name(name: str):
    """Retourne le mois (1-12) depuis le premier YYYYMMDD trouvé dans le nom; None si pas trouvé."""
    m = re.search(r"(\d{8})", name)  # cherche YYYYMMDD
    if not m:
        return None
    yyyymmdd = m.group(1)
    return int(yyyymmdd[4:6])

###############################################################################
# FUSION
###############################################################################

all_results = []
selected_pkls = []
skipped_pkls = []

# 1) Lister tous les fichiers dans tous les répertoires
all_pkls = []
for d in result_dirs:
    if not d.exists():
        print(f"[WARN] Dossier n'existe pas: {d}")
        continue
    all_pkls.extend(sorted(d.glob("gridsearch_results_*.pkl")))

print(f"Trouvé {len(all_pkls)} fichiers gridsearch_results_*.pkl au total.")

# 2) Appliquer le filtre de mois (si exclude_months non vide)
for p in all_pkls:
    mon = extract_month_from_name(p.name)
    if mon is not None and mon in exclude_months:
        skipped_pkls.append(p)
    else:
        selected_pkls.append(p)

print(f"Fichiers pris: {len(selected_pkls)} | ignorés (mois exclus): {len(skipped_pkls)}")
if skipped_pkls:
    for p in skipped_pkls:
        print(f"  - ignoré: {p.name}")

# 3) Charger et fusionner
for pkl_file in selected_pkls:
    with open(pkl_file, "rb") as f:
        try:
            data = pickle.load(f)
        except Exception as e:
            print(f"[ERREUR] Lecture {pkl_file}: {e}")
            continue

        # Cas typique: list de dict
        if isinstance(data, list):
            all_results.extend(data)
        # Par sécurité: si c'est un DataFrame (au cas où tu aurais changé le format un jour)
        else:
            # On essaie de gérer proprement:
            try:
                import pandas as pd
                if isinstance(data, pd.DataFrame):
                    all_results.extend(data.to_dict("records"))
                else:
                    # fallback: on ajoute tel quel
                    all_results.append(data)
            except ImportError:
                # si pandas pas dispo, on push juste l'objet
                all_results.append(data)

print(f"Fusion terminée! {len(all_results)} runs au total.")

# 4) Sauvegarde
with open(output_file, "wb") as f:
    pickle.dump(all_results, f)

print(f"Résultats fusionnés sauvegardés dans: {output_file.resolve()}")
