from flask import Flask, jsonify, request
from flask_cors import CORS
import joblib

# Initialize Flask app
app = Flask(__name__)

# Enable CORS only for the allowed domains (for browser-based requests)
CORS(app)


# Load pre-trained models and CountVectorizer
spam_model = joblib.load('spam-detection/nb_spam_model.pkl')
spam_cv = joblib.load('spam-detection/count_vectorizer.pkl')

# Define the route for prediction
@app.route('/api/predict', methods=['POST'])
def predict():

    data = request.get_json()

    if 'email' not in data:
        return jsonify({'error': 'Email text is required'}), 400

    email_text = data['email']

    # Preprocess the email text
    email_features = spam_cv.transform([email_text])

    # Predict using the pre-trained model
    prediction = spam_model.predict(email_features)

    # Return the result as a JSON response
    result = 'Spam' if prediction[0] == 1 else 'Not Spam'
    
    print(result)
    return jsonify({'prediction': result})

if __name__ == "__main__":
    app.run(host='0.0.0.0' , port=5000)
    
   
