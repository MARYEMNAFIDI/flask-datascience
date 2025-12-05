from flask import Flask, render_template, request, redirect, url_for
import os
import joblib
import pandas as pd

app = Flask(__name__, static_folder='static', template_folder='templates')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model', 'housing_model.pkl')

def load_model():
    if os.path.exists(MODEL_PATH):
        data = joblib.load(MODEL_PATH)
        # model file may contain dict {'model': pipeline, 'features': [...]}
        if isinstance(data, dict) and 'model' in data and 'features' in data:
            return data['model'], data['features']
        # else assume it's a bare pipeline trained on previous schema
        return data, None
    return None, None

model, FEATURES = load_model()
if FEATURES is None:
    # fallback features if model was previously saved without feature list
    FEATURES = ['surface','chambres','salles_de_bains','floor','ascenseur','terrasse','parking','type_propriete','quartier']

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        # Build inputs dict according to FEATURES expected by the model
        inputs = {}
        errors = []
        try:
            for feat in FEATURES:
                if feat in ['surface', 'chambres', 'salles_de_bains', 'floor']:
                    # numeric - these can be 0
                    val = request.form.get(feat, '')
                    if val == '' and feat in ['surface', 'chambres', 'salles_de_bains']:
                        errors.append(f"{feat} est requis")
                    inputs[feat] = float(val) if val != '' else 0.0
                elif feat in ['ascenseur', 'terrasse', 'parking']:
                    # boolean checkboxes
                    inputs[feat] = 1 if request.form.get(feat) else 0
                elif feat in ['type_propriete', 'quartier']:
                    # categorical - MUST be provided
                    val = request.form.get(feat, '').strip()
                    if not val:
                        errors.append(f"{feat} est requis")
                    inputs[feat] = val
                else:
                    # Other categorical
                    inputs[feat] = request.form.get(feat, '').strip()
            
            if errors:
                return render_template('index.html', error=' | '.join(errors))
                
        except ValueError as e:
            return render_template('index.html', error=f'Erreur de valeur: {str(e)}')

        if model is None:
            return render_template('result.html', error='Modèle non trouvé. Lancez python model/train_model.py')

        # Create DataFrame with exact features in correct order
        df_in = pd.DataFrame([inputs])
        # Ensure all features are present
        for feat in FEATURES:
            if feat not in df_in.columns:
                df_in[feat] = 0
        # Reorder columns to match model exactly
        df_in = df_in[FEATURES]
        
        try:
            pred = model.predict(df_in)[0]
            pred_rounded = int(round(pred))
        except Exception as e:
            return render_template('result.html', error=f'Erreur de prédiction: {str(e)}')
            
        return render_template('result.html', prediction=pred_rounded, inputs=inputs)

    return render_template('index.html')

@app.route('/train')
def train():
    train_script = os.path.join(os.path.dirname(__file__), 'model', 'train_model.py')
    if os.path.exists(train_script):
        import subprocess
        subprocess.check_call([os.sys.executable, train_script])
        global model, FEATURES
        model, FEATURES = load_model()
        # if model saved as dict, load_model returns pipeline and features list; ensure FEATURES contains 'quartier'
        if FEATURES is not None and 'quartier' not in FEATURES:
            FEATURES.append('quartier')
        return redirect(url_for('index'))
    return 'Script de training introuvable', 404

if __name__ == '__main__':
    app.run(debug=True)
