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

def getTopSortedSimilarityMatches(input_text, target_list):
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
        if sorted_similarities[i] > 0:
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
    content = request.json
    input_text = content['text']
    log_list = [f'Input text recieved was "{input_text}"']

    category_json = getCategory(input_text)
    deducted_category = category_json['prediction']
    log_list.append(f'Input categorized by QueryTypeDetection : {category_json["prediction"]} with confidence: {category_json["confidence"]} (0=retail, 1=brand, 2=category)')

    df_offer_retailer = pd.read_csv('data/offer_retailer.csv')
    df_cat = pd.read_csv('data/categories.csv')
    df_brand_category = pd.read_csv('data/brand_category.csv')

    unique_retailers = list(set([r['RETAILER'] for i, r in df_offer_retailer.iterrows() if not isNaN(r['RETAILER'])]))
    unique_brands = list(set([r['BRAND'] for i, r in df_offer_retailer.iterrows() if (r['BRAND'] != r['RETAILER']) & (not isNaN(r['BRAND']))]))
    unique_categories = list(set([r['PRODUCT_CATEGORY'] for i, r in df_cat.iterrows() if not isNaN(r['PRODUCT_CATEGORY'])]))
    unique_categories += list(set([r['IS_CHILD_CATEGORY_TO'] for i, r in df_cat.iterrows() if not isNaN(r['IS_CHILD_CATEGORY_TO'])]))
    unique_offers = list(set([r['OFFER'] for i, r in df_offer_retailer.iterrows() if not isNaN(r['OFFER'])]))

    offer_list = []

    if deducted_category == 0:
        similarity_list = getTopSortedSimilarityMatches(input_text, unique_retailers)
        log_list.append(f'Similarity matches with retailers: {similarity_list}')
        if len(similarity_list) > 0 and (similarity_list[0]['cosine_similarity'] > 0.4 or similarity_list[0]['fuzzy_score'] > 0.7):
            offer_list = getTopSortedSimilarityMatches(input_text, list(df_offer_retailer[df_offer_retailer['RETAILER'] == similarity_list[0]['concept']]['OFFER']))
    elif deducted_category == 1:
        similarity_list = getTopSortedSimilarityMatches(input_text, unique_brands)
        log_list.append(f'Similarity matches with brands: {similarity_list}')
        if len(similarity_list) > 0 and (similarity_list[0]['cosine_similarity'] > 0.4 or similarity_list[0]['fuzzy_score'] > 0.7):
            offer_list = getTopSortedSimilarityMatches(input_text, list(df_offer_retailer[df_offer_retailer['BRAND'] == similarity_list[0]['concept']]['OFFER']))
    else:
        lst_syn = [getTopSortedSimilarityMatches(word, unique_categories) for word in [input_text] + getSynonyms(input_text)]
        flat_list = [elem for sublist in lst_syn for elem in sublist]
		# Remove duplicates
        max_similarity_dict = {}
        for item in flat_list:
            concept = item['concept']
            similarity = item['cosine_similarity']
            if concept not in max_similarity_dict or similarity > max_similarity_dict[concept]['cosine_similarity']:
                max_similarity_dict[concept] = item
		
		# Convert the dictionary values to a list to get the final, deduplicated list
        similarity_list = list(max_similarity_dict.values())
        similarity_list = [item for item in similarity_list if item['cosine_similarity'] > 0 or item['fuzzy_score'] > 0.85]
        similarity_list = sorted(similarity_list, key=lambda x: x['cosine_similarity'], reverse=True)
        log_list.append(f'Similarity matches with categories: {similarity_list}')

		# Get the offer list comparing directly to offers
        offer_list = [getTopSortedSimilarityMatches(word, unique_offers) for word in [input_text] + getSynonyms(input_text)]
        offer_list = [elem for sublist in offer_list for elem in sublist]
        offer_list = sorted(offer_list, key=lambda x: x['cosine_similarity'], reverse=True)

    # Prepare response
    response = {'offer_list':offer_list, 'LOG_LIST':log_list}
    return jsonify(response)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
