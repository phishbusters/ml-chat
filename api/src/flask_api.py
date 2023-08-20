from flask import Flask, request, jsonify
import boto3
import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode
import re
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
import scipy.sparse as sp
from io import StringIO

# Descargar stopwords si no existen
try:
    stop_words = set(stopwords.words('spanish'))
except:
    import nltk
    nltk.download('stopwords')
    stop_words = set(stopwords.words('spanish'))

ps = PorterStemmer()

def preprocess_text(text):
    text = re.sub(r'\$\d+|\d+\$', 'MONTOTOKEN', text)
    text = re.sub(r'http\S+|www\S+', 'URLTOKEN', text)
    text = unidecode(text)
    text = re.sub(r'[^\w\s]', '', text.lower())
    text = ' '.join([ps.stem(word) for word in text.split() if word not in stop_words])
    return text

# Configuración de S3
bucket_name = "phish-busters"
s3 = boto3.client('s3')

# Descargar y cargar modelos y vectorizador desde S3
s3.download_file(bucket_name, 'CHAT_MODEL.pkl', '/tmp/CHAT_MODEL.pkl')
s3.download_file(bucket_name, 'VECTORIZER_CHAT_MODEL.pkl', '/tmp/VECTORIZER_CHAT_MODEL.pkl')
model = joblib.load('/tmp/CHAT_MODEL.pkl')
tfidf_vectorizer = joblib.load('/tmp/VECTORIZER_CHAT_MODEL.pkl')

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict():
    # Extraer el mensaje y características opcionales desde el request
    message = request.json['message']
    additional_features = request.json.get('additional_features', [0, 0, 0, 0])

    # Preprocesar el mensaje
    processed_message = preprocess_text(message)

    # Transformar con TF-IDF
    message_tfidf = tfidf_vectorizer.transform([processed_message])

    # Combinar con las características adicionales
    combined_features = sp.hstack((message_tfidf, sp.csr_matrix([additional_features])))

    # Realizar la predicción
    prediction = model.predict(combined_features)
    confidence = model.predict_proba(combined_features)[0][prediction[0]]

    # Almacenar el mensaje en S3
    new_data = pd.DataFrame({
      'message': [message],
      'phishing': prediction[0],
      'url': None,
      'dominio': None,
      'tiene_https': additional_features[0],
      'longitud_url': additional_features[1],
      'presencia_subdominio': additional_features[2],
      'palabras_clave_sospechosas': additional_features[3],
      'confidence': [confidence]
    })

    csv_buffer = StringIO()
    new_data.to_csv(csv_buffer, index=False)
    s3_resource = boto3.resource('s3')
    s3_resource.Object(bucket_name, 'new_dataset_from_requests.csv').put(Body=csv_buffer.getvalue())

    # Devolver la respuesta
    response = {
        'prediction': "phishing" if prediction[0] == 1 else "no phishing",
        'confidence': confidence
    }
    
    return jsonify(response)

if __name__ == '__main__':
    app.run(debug=True)
