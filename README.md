📈 An Intelligent BTC Forecasting & Trading DashboardCe projet propose un tableau de bord interactif pour la prévision du prix du Bitcoin (BTC) en utilisant un modèle Temporal Fusion Transformer (TFT). L'application est entièrement conteneurisée avec Docker pour garantir une installation sans erreur et une compatibilité totale des versions.

🐳 Utilisation avec Docker (Recommandé)L'utilisation de Docker est la méthode la plus simple car elle contient déjà toutes les bibliothèques configurées (Python 3.10, Pandas 1.5.3, PyTorch, etc.).

une fois que le clone est fait, suivez instructions suivantes :
**Ouvrir le repertoire de l'application** : Commande bash : cd An-Intelligent-BTC-Forecasting-trading-dashboard
**Construction de l'image docker :** Commande bash : docker build -t btc-dashboard-final .
**Lancement  du dashboard :** Commande : docker run -p 8501:8501 btc-dashboard-final 
**Accès:** Ouvrez votre navigateur sur http://localhost:8501

📊 Gestion des DonnéesChargement par défautAu lancement, le dashboard charge automatiquement le fichier d'historique inclus :
👉 Val_dec_2025_Binance.csvUtiliser un autre historiqueIl est possible de charger un autre fichier d'historique directement depuis l'interface Streamlit pour analyser d'autres périodes.

⚠️ IMPORTANT : Pour que le modèle puisse générer des prédictions, votre fichier CSV doit respecter strictement la structure suivante :

Colonne "timestamp" type float : Temps au format UNIX (secondes)
Colonne "high" type float: Prix le plus haut
Colonne "close" type  float :Prix de clôture
Colonne "volume"type float float: Volume de transaction 
Colonne "low" type float : Prix le plus bas
Colonne "open" type float: Prix d'ouverture

**Spécifications techniques :**
Intervalle : Les données doivent avoir un pas de 1 minute (60 secondes entre chaque timestamp).
Format : Le fichier doit être un CSV avec séparateur virgule.

**package tree**
.
├── BTC_Dashboard_final.py     # Script principal Streamlit
├── TFT_model.ckpt        # Le modèle entraîné (indispensable)
├── requirements.txt      # Dépendances figées
├── Dockerfile            # Configuration du conteneur
├── Val_dec_2025_Binance.csv # Données d'exemple
└── .dockerignore         # Fichiers exclus du build

🛠️ Stack Technique :
Deep Learning : PyTorch Forecasting (TFT Model)
Interface : Streamlit
Visualisation : Plotly
Conteneurisation : Docker (Image Python-slim)


📝 Notes sur le ModèleLe modèle (TFT_model.ckpt) utilise des mécanismes d'attention temporelle pour capturer les tendances du marché. Le code inclut des correctifs de compatibilité pour assurer la lecture du modèle sur les environnements récents.Développé par tourki23