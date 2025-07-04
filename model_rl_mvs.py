#!/usr/bin/python

import pandas as pd
import numpy as np
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import joblib
import sys
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer
from nltk import pos_tag
import joblib
from flask import Flask
from flask_restx import Api, Resource, fields

import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('averaged_perceptron_tagger')
nltk.download('averaged_perceptron_tagger_eng')

# Inicialización
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# Función para mapear 
def get_wordnet_pos(tag):
    if tag.startswith('J'):
        return wordnet.ADJ
    elif tag.startswith('V'):
        return wordnet.VERB
    elif tag.startswith('N'):
        return wordnet.NOUN
    elif tag.startswith('R'):
        return wordnet.ADV
    else:
        return wordnet.NOUN


search = joblib.load(os.path.dirname(__file__) + '/rl_mvs_gnr.pkl')

vectorizer = joblib.load(os.path.dirname(__file__) + '/vectr.pkl')


app = Flask(__name__)
api = Api(
    app,
    version='1.0',
    title='Movies Genres Prediction API',
    description='API para predicción con múltiples features'
)

ns = api.namespace('predict', description='Clasificador de generos')

# Configurar los parametros de entrada
parser = ns.parser()
parser.add_argument('year', type=int, required=True, help='year', location='args')
parser.add_argument('title', type=str, required=True, help='title', location='args')
parser.add_argument('plot', type=str, required=True, help='plot', location='args')
parser.add_argument('rating', type=float, required=True, help='rating', location='args')

# Modelo para la respuesta en formato variable-valor
prediction_model = api.model('Prediccion', {
    'variable': fields.String,
    'valor': fields.String
})

resource_fields = api.model('Resultado', {
    'predicciones': fields.List(fields.Nested(prediction_model))
})


@ns.route('/')
class genrmoviesApi(Resource):
    @ns.doc(parser=parser)
    @ns.marshal_with(resource_fields)
    def get(self):
        args = parser.parse_args()

        def preprocess_text_full(text):
            tokens = word_tokenize(text.lower())
            tagged_tokens = pos_tag(tokens)
            
            processed = []
            for word, tag in tagged_tokens:
                if word.isalpha() and word not in stop_words:
                    pos = get_wordnet_pos(tag)
                    lemma = lemmatizer.lemmatize(word, pos=pos)
                    processed.append(lemma)
            
            return ' '.join(processed)
        
        # Crear DataFrame con los parámetros recibidos
        input_data = {
            'year': [args['year']],
            'title': [args['title']],
            'plot': [args['plot']],
            'rating': [args['rating']]
        }
        
        df = pd.DataFrame(input_data)

        df['text_combined'] = df['title']  + ' ' + df['plot']

        df['text_clean'] = df['text_combined'].apply(preprocess_text_full)

        X_test_dtm = vectorizer.transform(df['text_clean'])

        result_df = search.predict_proba(X_test_dtm)

        cols = ['p_Action', 'p_Adventure', 'p_Animation', 'p_Biography', 'p_Comedy', 'p_Crime', 'p_Documentary', 'p_Drama', 'p_Family',
        'p_Fantasy', 'p_Film-Noir', 'p_History', 'p_Horror', 'p_Music', 'p_Musical', 'p_Mystery', 'p_News', 'p_Romance',
        'p_Sci-Fi', 'p_Short', 'p_Sport', 'p_Thriller', 'p_War', 'p_Western']
        
        predictions = pd.DataFrame(result_df, index=df.index, columns=cols)

        predictions_list = []
        for col in cols:
            for index, row in predictions.iterrows():
                predictions_list.append({
                    'variable': col,
                    'valor': str(row[col])
                })
                
        return {'predicciones': predictions_list}, 200

app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
        