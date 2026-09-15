"""
Mini-projet : Classification des vins (Wine dataset)
Auteur : Ivan
Objectif : Prédire la catégorie d'un vin (1, 2 ou 3) à partir de 13 mesures
physico-chimiques, en utilisant un arbre de décision (Decision Tree).
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# 1. Chargement des données
df = pd.read_csv("wine.csv")

# 2. Préparation des variables
X = df.drop("class", axis=1)
y = df["class"]

# 3. Split train/test (80/20)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# 4. Entraînement du modèle (arbre de décision, profondeur limitée à 3)
model = DecisionTreeClassifier(max_depth=3)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)

# 5. Évaluation
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy : ", accuracy)
print("Confusion matrix :\n", confusion_matrix(y_test, y_pred))
print("Classification report :\n", classification_report(y_test, y_pred))

# 6. Importance des variables
importances = pd.Series(model.feature_importances_, index=X.columns)
importances = importances.sort_values(ascending=False)
print("Importance des variables :\n", importances)

importances.plot(kind="bar", figsize=(10, 5))
plt.title("Importance des variables")
plt.ylabel("Importance")
plt.tight_layout()
plt.show()

# 7. Comparaison Train vs Test selon la profondeur (détection de surapprentissage)
for depth in [3, 6]:
    m = DecisionTreeClassifier(max_depth=depth)
    m.fit(X_train, y_train)
    train_acc = accuracy_score(y_train, m.predict(X_train))
    test_acc = accuracy_score(y_test, m.predict(X_test))
    print(f"max_depth={depth} -> Train: {train_acc:.4f} | Test: {test_acc:.4f}")

# 8. Visualisation de l'arbre final (max_depth=3, retenu pour sa simplicité
# et son interprétabilité, à performance de test égale)
plt.figure(figsize=(20, 10))
plot_tree(model, feature_names=X.columns, class_names=["1", "2", "3"], filled=True, rounded=True)
plt.show()
