from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from flask import Flask, request, jsonify
from fuzzywuzzy import fuzz, process
from nltk.stem import PorterStemmer
from nltk.corpus import wordnet
from textblob import TextBlob
from flask_cors import CORS
import pandas as pd
import numpy as np
import requests
import json
import nltk
import re

nltk.download('wordnet')
nltk.download('punkt')


app = Flask(__name__)
CORS(app)

def isNaN(num):
    return num != num

def getSynonyms(word):
    synonyms = []
    for syn in wordnet.synsets(word):
        for lemma in syn.lemmas():
            synonyms.append(lemma.name().replace('_', ' '))
    # Remove duplicates
    synonyms = list(set(synonyms))
    return synonyms

def listToJson(string_list):
    json_list = []
    for item in string_list:
        json_object = {
            'concept': item,
            'cosine_similarity': -1,
            'fuzzy_score': -1
        }
        json_list.append(json_object)
    return json_list

def tokenize(text):
    tokens = nltk.word_tokenize(text)
    stemmer = PorterStemmer()
    return [stemmer.stem(t) for t in tokens]

def getTopSortedSimilarityMatches(input_text, target_list, allow_zero = False):
    '''
    Returns the top sorted matches of the list with respect to the input text (TF-IDF -> cosine similarity) if similarity > 0,
    If no matches are found, it triew using 'fuzzy' matches to compensate for misspelling.
    '''
    documents = [input_text] + target_list
    tfidf_vectorizer = TfidfVectorizer(tokenizer=tokenize, ngram_range=(1, 3))

    tfidf_matrix = tfidf_vectorizer.fit_transform(documents)
    cosine_similarities = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:]).flatten()

    sorted_index = np.argsort(-cosine_similarities)
    sorted_retailers = [target_list[i] for i in sorted_index]
    sorted_similarities = [cosine_similarities[i] for i in sorted_index]

    lst_resp = []

    for i, retailer in enumerate(sorted_retailers):
        if sorted_similarities[i] > 0 or allow_zero:
            lst_resp.append({'concept': retailer, 'cosine_similarity': sorted_similarities[i], 'fuzzy_score' : 0})
    
    # If the list is empty, try using fuzzy matching as a fallback
    if not lst_resp:
        highest = process.extractOne(input_text, target_list)
        if highest:
            lst_resp.append({'concept': highest[0], 'cosine_similarity':0, 'fuzzy_score': highest[1]/100.0})
    
    return lst_resp

def getCategory(text_input):
    '''
    This function calls our other 'QueryTypeDetector' service and returns the category and confidence.
    '''
    url = "http://20.51.242.109:5000/predict"
    data = {"text": text_input}
    headers = {"Content-Type": "application/json"}
    
    response = requests.post(url, data=json.dumps(data), headers=headers)
    #print("Status Code:", response.status_code)
    return response.json()


@app.route('/get_offers', methods=['POST'])
def getOffers():
    # Get input from JSON request and log it
    content = request.json
    input_text = content['text']
    log_list = [f'Input text received was "{input_text}"']

    # Categorize the input
    category_json = getCategory(input_text)
    deducted_category = category_json['prediction']
    log_list.append(f'Input categorized by QueryTypeDetection: {deducted_category} with confidence: {category_json["confidence"]}')

    # Load data from CSV files
    df_offer_retailer = pd.read_csv('data/offer_retailer.csv')
    df_cat = pd.read_csv('data/categories.csv')

    # Extract unique retailers, brands, categories, and offers
    unique_retailers = df_offer_retailer['RETAILER'].dropna().unique().tolist()
    unique_brands = df_offer_retailer.loc[df_offer_retailer['BRAND'] != df_offer_retailer['RETAILER'], 'BRAND'].dropna().unique().tolist()
    unique_categories = df_cat['PRODUCT_CATEGORY'].dropna().unique().tolist() + df_cat['IS_CHILD_CATEGORY_TO'].dropna().unique().tolist()
    unique_offers = df_offer_retailer['OFFER'].dropna().unique().tolist()

    offer_list = []

    # Define a condition to filter out low similarity scores
    def isHighSimilarity(item):
        return item['cosine_similarity'] > 0.4 or item['fuzzy_score'] > 0.7

    if deducted_category == 0:
        similarity_list = getTopSortedSimilarityMatches(input_text, unique_retailers)
        log_list.append(f'Similarity matches with retailers: {similarity_list}')
        if similarity_list and isHighSimilarity(similarity_list[0]):
            top_retailer = similarity_list[0]['concept']
            offer_list = getTopSortedSimilarityMatches(input_text, df_offer_retailer[df_offer_retailer['RETAILER'] == top_retailer]['OFFER'].tolist(), True)

    elif deducted_category == 1:
        similarity_list = getTopSortedSimilarityMatches(input_text, unique_brands)
        log_list.append(f'Similarity matches with brands: {similarity_list}')
        if similarity_list and isHighSimilarity(similarity_list[0]):
            top_brand = similarity_list[0]['concept']
            offer_list = getTopSortedSimilarityMatches(input_text, df_offer_retailer[df_offer_retailer['BRAND'] == top_brand]['OFFER'].tolist(), True)

    else:
        words = [input_text] + getSynonyms(input_text)
        similarity_list = [getTopSortedSimilarityMatches(word, unique_categories) for word in words]
        similarity_list = [item for sublist in similarity_list for item in sublist]

        # Remove duplicates by keeping the highest similarity score for each concept
        max_similarity_dict = {}
        for item in similarity_list:
            concept, similarity = item['concept'], item['cosine_similarity']
            max_similarity_dict.setdefault(concept, item).update(cosine_similarity=max(similarity, max_similarity_dict.get(concept, {}).get('cosine_similarity', 0)))

        similarity_list = sorted(max_similarity_dict.values(), key=lambda x: x['cosine_similarity'], reverse=True)
        log_list.append(f'Similarity matches with categories: {similarity_list}')

        offer_list = [getTopSortedSimilarityMatches(word, unique_offers) for word in words]
        offer_list = [item for sublist in offer_list for item in sublist]
        offer_list = sorted(offer_list, key=lambda x: x['cosine_similarity'], reverse=True)

    # Prepare response
    response = {'offer_list': offer_list, 'LOG_LIST': log_list}
    return jsonify(response)


if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
