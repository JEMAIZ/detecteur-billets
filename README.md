# 🛡️ Détecteur de Faux Billets - Expertise ONCFM

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![Scikit-Learn](https://img.shields.io/badge/scikit--learn-%23F7931E.svg?style=for-the-badge&logo=scikit-learn&logoColor=white)

Cette application web interactive permet de détecter les contrefaçons de billets de banque en se basant sur leurs dimensions géométriques. Développée pour l'Organisation Nationale de Lutte contre la Falsification Monétaire (ONCFM), elle utilise des modèles de Machine Learning pour automatiser l'expertise.

## 🚀 Fonctionnalités

- **Imputation Automatisée** : Gestion des données manquantes (marge inférieure) par régression linéaire.
- **Multi-Modèles** : Comparaison des prédictions entre KNN, Régression Logistique et Random Forest.
- **Visualisations Avancées** :
    - **Distribution Bivariée** : Analyse de la densité (Margin Low vs Length) avec courbes KDE.
    - **Projection PCA** : Visualisation des billets dans un espace réduit pour identifier les clusters.
- **Expertise de Masse** : Chargement de fichiers CSV et export des résultats avec indices de confiance.

## 📁 Structure du Projet

- `app.py` : Code principal de l'interface Streamlit.
- `scaler.pkl` : Normalisation des données (StandardScaler).
- `pca.pkl` : Modèle de réduction de dimensionnalité.
- `knn.pkl` / `logisticregression.pkl` : Modèles prédictifs entraînés.
- `requirements.txt` : Liste des dépendances pour le déploiement Cloud.

## 🛠️ Installation et Utilisation Locale

1. **Cloner le projet**
   ```bash
   git clone [https://github.com/VOTRE_NOM/detecteur-billets.git](https://github.com/VOTRE_NOM/detecteur-billets.git)
   cd detecteur-billets
2. **Installer les dépendances**  pip install -r requirements.txt
3. **Lancer l'application** streamlit run app.py
