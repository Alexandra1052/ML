from flask import Flask, request, jsonify
import joblib
import pandas as pd
from flask_cors import CORS


model = joblib.load('model/model.pkl')
le = joblib.load('model/label_encoder.pkl')

app = Flask(__name__)
CORS(app, resources={r"/predict": {"origins": "*"}}, supports_credentials=True)

@app.route('/predict', methods=['POST', 'OPTIONS'])
def predict():
    if request.method == 'OPTIONS':
      
        return '', 200

    data = request.get_json()

    if 'score' not in data or 'profile' not in data:
        return jsonify({"error": "Missing 'score' or 'profile' in request"}), 400
    
    score = data['score']
    profile = data['profile']
    
    print(f"Received score: {score}, profile: {profile}")
    
    try:
        profile_code = le.transform([profile])[0]
        print(f"Profile code: {profile_code}")
    except ValueError:
        return jsonify({"error": f"Profile '{profile}' is invalid"}), 400
    
    licee = pd.read_csv('data/medii_bucuresti.csv', delimiter=';')
    print(f"Licee data loaded: {licee.head()}")

    licee['Profil_Code'] = le.transform(licee['Profil'])
    input_data = licee[['Profil_Code', 'Ultima_Medie']]

    predictions = model.predict(input_data)
    print(f"Predictions: {predictions}")

    licee['Predicted_Medie'] = predictions

    # Creează mesaje personalizate pentru fiecare liceu
    def generate_message(predicted_score, user_score):
        if user_score < predicted_score:
            points_needed = predicted_score - user_score
            return f"You need {points_needed:.2f} more points to reach {predicted_score:.2f} at this school."
        elif user_score > predicted_score:
            points_above = user_score - predicted_score
            return f"You're above the required score by {points_above:.2f} points! Keep up the good work!"
        else:
            return "You are right on target! Good luck!"

    licee['Message'] = [generate_message(row['Predicted_Medie'], score) for _, row in licee.iterrows()]

    # Selectează top 3 licee cele mai apropiate ca medie
    licee['diff'] = abs(licee['Predicted_Medie'] - score)
    top_licee = licee.nsmallest(3, 'diff')[['Liceu', 'Profil', 'Predicted_Medie', 'Message']]

    return jsonify(top_licee.to_dict(orient='records'))


if __name__ == '__main__':
    app.run(debug=True)
