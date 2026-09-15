# 🍷 Classification des vins (Wine Dataset)

Mini-projet de Machine Learning : prédire la catégorie d'un vin (1, 2 ou 3) à partir de 13 mesures physico-chimiques, à l'aide d'un arbre de décision (Decision Tree).

## 📋 Contexte

Projet réalisé dans le cadre de mon apprentissage du Machine Learning (cursus Intelligence Artificielle, L1), pour mettre en pratique les arbres de décision sur un cas réaliste multi-classe et multi-variables.

## 📊 Dataset

Le dataset `wine.csv` contient 178 vins et 13 variables physico-chimiques :

`alcohol`, `malic_acid`, `ash`, `alcalinity_of_ash`, `magnesium`, `total_phenols`, `flavanoids`, `nonflavanoid_phenols`, `proanthocyanins`, `color_intensity`, `hue`, `od280_od315`, `proline`

La variable cible `class` indique la catégorie du vin (1, 2 ou 3).

## 🔧 Méthodologie

1. **Préparation** : séparation features (X, 13 variables) / cible (y, `class`)
2. **Split** train/test (80/20)
3. **Modélisation** : arbre de décision (`DecisionTreeClassifier`)
4. **Évaluation** : accuracy, matrice de confusion 3×3, classification report par classe
5. **Analyse de l'importance des variables** (`feature_importances_`)
6. **Détection du surapprentissage** : comparaison accuracy train vs test à différentes profondeurs (`max_depth`)
7. **Visualisation** de l'arbre final (`plot_tree`)

## 📈 Résultats

- **Accuracy** (max_depth=3) : 0.9444 (34/36 sur le jeu de test)
- **Variables les plus déterminantes** : `flavanoids`, `color_intensity`, `proline`
- **Surapprentissage** : à `max_depth=6`, l'accuracy sur train atteint 1.0 sans gain sur le test (toujours 0.9444) — `max_depth=3` est donc préféré, à performance égale, pour sa simplicité et son interprétabilité (rasoir d'Occam)

## 🛠️ Technologies

- Python
- pandas
- scikit-learn
- matplotlib

## 🚀 Utilisation

```bash
pip install -r requirements.txt
python wine_classification.py
```

## 📁 Structure du projet

```
├── wine_classification.py   # Script principal
├── wine.csv                  # Dataset
├── requirements.txt          # Dépendances
└── README.md                 # Ce fichier
```

## ✍️ Auteur

Ivan — Étudiant L1 Intelligence Artificielle
