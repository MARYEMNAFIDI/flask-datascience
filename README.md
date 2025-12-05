# Estimation Immobilière Casablanca – Flask

Application Flask moderne pour prédire le prix d'un bien immobilier à Casablanca (Maroc) avec un modèle Machine Learning.

## Structure du projet

```
flask-datascience/
│
├── app.py                  # Application Flask principale
├── model/
│   ├── train_model.py      # Script d'entraînement du modèle (RandomForest)
│   └── housing_model.pkl   # Modèle ML entraîné (généré)
│
├── templates/
│   ├── base.html           # Template principal (dark, gradient)
│   ├── index.html          # Formulaire d'entrée (UI paysage, français)
│   └── result.html         # Page résultat (prix estimé)
│
├── static/
│   ├── css/style.css       # Styles personnalisés
│   └── img/house.jpg       # Image décorative (optionnelle)
│
├── requirements.txt
├── data/
│   └── housing_data.csv    # Dataset Maroc (Kaggle)
└── README.md
```

## Installation & utilisation

1. **Activer l'environnement virtuel**
   ```powershell
   & "C:\Users\pc gold\flask-datascience\venv\Scripts\Activate.ps1"
   ```
2. **Installer les dépendances**
   ```powershell
   pip install -r requirements.txt
   ```
3. **Télécharger le dataset**
   - Kaggle : [Houses Prices in Morocco](https://www.kaggle.com/datasets/yassinesadiki/housing-in-morocco)
   - Placer le fichier `housing_data.csv` dans le dossier `data/`
4. **Entraîner le modèle**
   ```powershell
   python model/train_model.py
   ```
   - Le modèle est entraîné sur Casablanca uniquement (filtre automatique dans le script)
   - Les quartiers de Casablanca sont extraits automatiquement
5. **Lancer l'application Flask**
   ```powershell
   python app.py
   ```
6. **Accéder à l'interface**
   - Ouvrir [http://127.0.0.1:5000/](http://127.0.0.1:5000/) dans le navigateur

## Fonctionnalités principales

- **Formulaire moderne paysage** : tous les labels en français, dropdown pour les quartiers de Casablanca
- **Prédiction fiable** : modèle RandomForest, pipeline scikit-learn, bug d'affichage corrigé
- **UI dark & responsive** : design professionnel, gradient, cartes glassmorphism
- **Résultat clair** : prix affiché en MAD, détails du bien, commodités

## Notes techniques

- Le modèle utilise les features : surface, chambres, salles_de_bains, étage, ascenseur, terrasse, parking, type_propriete, quartier
- Le formulaire propose une liste déroulante des quartiers extraits du dataset
- Les valeurs sont validées côté serveur (aucun champ obligatoire vide)
- Le bug d'affichage du prix (0 MAD) est corrigé : le prix affiché correspond bien à la prédiction du modèle

## Personnalisation

- Pour changer la ville ou les quartiers, modifier le script `model/train_model.py`
- Pour adapter le style, modifier `templates/index.html` et `static/css/style.css`

---

**MARYEM NAFIDI** : Projet pédagogique, adapté pour le marché marocain.

