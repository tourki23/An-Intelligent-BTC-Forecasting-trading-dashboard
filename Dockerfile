# 1. Utilisation de l'image Python stable et légère
FROM python:3.10-slim

# 2. Variables d'environnement pour optimiser Python dans Docker
# Évite que Python n'écrive des fichiers .pyc (gain de place)
ENV PYTHONDONTWRITEBYTECODE=1
# Force l'affichage des logs en temps réel sans mise en tampon
ENV PYTHONUNBUFFERED=1

# 3. Installation des dépendances système indispensables
# On ne garde que build-essential pour la compilation des libs de Data Science
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# 4. Définition du répertoire de travail dans le conteneur
WORKDIR /app

# 5. Gestion des dépendances Python
# On copie d'abord le fichier requirements pour profiter du cache Docker
COPY requirements.txt .

# Installation des packages (cette étape sera longue à cause de torch)
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# 6. Copie du reste des fichiers du projet
# Le .dockerignore s'occupera d'exclure ton .venv local
COPY . .

# 7. Exposition du port utilisé par Streamlit
EXPOSE 8501

# 8. Commande de lancement automatique du Dashboard
# L'adresse 0.0.0.0 est cruciale pour l'accès externe via Docker
ENTRYPOINT ["streamlit", "run", "BTC_Dashboard_final.py", "--server.port=8501", "--server.address=0.0.0.0"]