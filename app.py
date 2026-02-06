import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import plotly.express as px
from sklearn.linear_model import LinearRegression

# --- CORRECTIFS DE COMPATIBILITÉ ---
pd.DataFrame.iteritems = pd.DataFrame.items 

# --- CONFIGURATION DE LA PAGE ---
st.set_page_config(page_title="ONCFM - Détection de Faux Billets", layout="wide")

# --- CHARGEMENT DES ASSETS (MODÈLES, SCALER, PCA) ---
@st.cache_resource
def load_assets(model_name):
    try:
        scaler = joblib.load("scaler.pkl")
        pca = joblib.load("pca.pkl")
        # Chargement dynamique du modèle sélectionné
        filename = f"{model_name.lower().replace(' ', '')}.pkl"
        model = joblib.load(filename)
        return model, scaler, pca
    except Exception as e:
        st.error(f"Erreur de chargement des actifs : {e}")
        st.info("Assurez-vous que scaler.pkl, pca.pkl et les modèles (.pkl) sont dans le répertoire.")
        st.stop()

# --- FONCTION D'IMPUTATION (Régression Linéaire) ---
def handle_imputation(df):
    features_reg = ['diagonal', 'height_left', 'height_right', 'margin_up', 'length']
    if df['margin_low'].isnull().sum() > 0:
        df_train = df.dropna(subset=['margin_low'])
        df_pred = df[df['margin_low'].isnull()]
        if len(df_train) > 0:
            reg = LinearRegression()
            reg.fit(df_train[features_reg], df_train['margin_low'])
            df.loc[df['margin_low'].isnull(), 'margin_low'] = reg.predict(df_pred[features_reg])
    return df

# --- INTERFACE PRINCIPALE ---
def main():
    st.title("🛡️ Système Expert de Lutte contre la Contrefaçon")
    st.markdown("Analyse géométrique et détection d'authenticité par Machine Learning.")

    # --- SIDEBAR : CONFIGURATION ---
    st.sidebar.header("⚙️ Configuration")
    
    model_choice = st.sidebar.selectbox(
        "Modèle de prédiction",
        ["KNN", "Logistic Regression", "Random Forest"]
    )
    
    model, scaler, pca = load_assets(model_choice)
    st.sidebar.success(f"Modèle actif : {model_choice}")

    uploaded_file = st.sidebar.file_uploader("Charger un fichier de test (CSV)", type="csv")
    separator = st.sidebar.selectbox("Séparateur", [";", ","])

    if uploaded_file is not None:
        df_raw = pd.read_csv(uploaded_file, sep=separator)
        required_features = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']

        if not set(required_features).issubset(df_raw.columns):
            st.error(f"Colonnes manquantes. requis : {required_features}")
            return

        # Traitement des données
        df_clean = handle_imputation(df_raw.copy())

        # --- ONGLETS ---
        tab1, tab2 = st.tabs(["📊 Analyse Exploratoire & PCA", "🔍 Résultats des Prédictions"])

        with tab1:
            col1, col2 = st.columns([1, 1])
            
            
            with col1:
                st.subheader("Distribution : Margin Low vs Length")
                
                # On redéfinit explicitement les colonnes pour éviter le NameError
                features_list = ['diagonal', 'height_left', 'height_right', 'margin_low', 'margin_up', 'length']
                
                # 1. Préparation des données
                df_dist = df_clean.copy()
                # On utilise .values pour éviter le ValueError précédent
                X_scaled_dist = scaler.transform(df_clean[features_list].values)
                df_dist['Verdict'] = ["Authentique" if p == 1 else "Contrefaçon" for p in model.predict(X_scaled_dist)]
                
                # 2. Création du graphique avec Seaborn (JointGrid)
                # Utilisation de margin_low et length
                plt.figure(figsize=(10, 8))
                
                try:
                    g = sns.JointGrid(
                        data=df_dist, 
                        x="margin_low", 
                        y="length", 
                        hue="Verdict", 
                        palette={'Authentique': '#2ecc71', 'Contrefaçon': '#e74c3c'}
                    )
                    
                    # Centre : Nuage de points
                    g.plot_joint(sns.scatterplot, alpha=0.6, s=60)
                    
                    # Côtés : Courbes de densité (KDE) pour voir les "bosses" de distribution
                    g.plot_marginals(sns.kdeplot, fill=True, alpha=0.4)
                    
                    # Réglages des axes
                    g.ax_joint.set_xlabel("Marge Inférieure (mm)")
                    g.ax_joint.set_ylabel("Longueur du billet (mm)")
                    
                    st.pyplot(g.fig)
                    
                except Exception as e:
                    st.error(f"Erreur lors de la génération du graphique : {e}")
                
                st.info("""
                **Analyse visuelle :**
                * Les **Authentiques** (verts) ont tendance à être plus longs et à avoir une marge basse plus petite.
                * Les **Contrefaçons** (rouges) sont souvent plus courts avec une marge basse plus importante.
                * Les courbes en haut et à droite montrent la séparation nette entre les deux groupes.
                """)
            with col2:
                st.subheader("Projection sur le plan factoriel (PCA)")
                # Préparation PCA
                X_scaled = scaler.transform(df_clean[required_features].values)
                coords = pca.transform(X_scaled)
                
                # DataFrame pour Plotly
                df_pca = pd.DataFrame(coords, columns=['PC1', 'PC2'])
                df_pca['ID'] = df_raw['id'] if 'id' in df_raw.columns else df_raw.index
                df_pca['Verdict'] = ["Vrai" if p == 1 else "Faux" for p in model.predict(X_scaled)]

                fig_pca = px.scatter(
                    df_pca, x='PC1', y='PC2', color='Verdict',
                    hover_data=['ID'],
                    color_discrete_map={'Vrai': '#2ecc71', 'Faux': '#e74c3c'},
                    title="Visualisation interactive des billets"
                )
                fig_pca.add_hline(y=0, line_dash="dash", line_color="grey")
                fig_pca.add_vline(x=0, line_dash="dash", line_color="grey")
                st.plotly_chart(fig_pca, use_container_width=True)
                st.info("💡 Survolez les points pour identifier l'ID des billets suspects.")

        with tab2:
            st.subheader(f"Prédictions via {model_choice}")
            
            # Prédictions finales
            X_final = scaler.transform(df_clean[required_features].values)
            y_pred = model.predict(X_final)
            y_probs = model.predict_proba(X_final)

            df_raw['Résultat'] = ["Authentique" if p == 1 else "Contrefaçon" for p in y_pred]
            df_raw['Confiance (%)'] = np.round(np.max(y_probs, axis=1) * 100, 2)

            # Affichage du tableau
            st.dataframe(df_raw)

            # Résumé en colonnes
            c1, c2 = st.columns(2)
            with c1:
                fig_pie, ax_pie = plt.subplots()
                df_raw['Résultat'].value_counts().plot.pie(autopct='%1.1f%%', colors=['#2ecc71', '#e74c3c'], ax=ax_pie)
                ax_pie.set_ylabel('')
                st.pyplot(fig_pie)
            
            with c2:
                st.write("### Actions")
                csv = df_raw.to_csv(index=False, sep=";").encode('utf-8')
                st.download_button("📥 Télécharger le rapport complet", csv, "expert_report.csv", "text/csv")

    else:
        st.info("Veuillez charger un fichier CSV pour débuter l'expertise.")

if __name__ == "__main__":
    main()