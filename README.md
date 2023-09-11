# Transformer Based IR-Deployment
### Context
This platform recieves a text as input and returns a list of offers that are more relevant to that text performing an Information Retrieval algorithm that incudes a multi class text classification transformer based model, TF-IDF, cosine similarity, stemming and lemmatization, "fuzzy" similarity and  data augmentation through synonym generation.

### Example:
input : {"text" : "sams club membership"} <br>
output : {"offer_list":[{"concept":"Spend $50 on a Full-Priced new Club Membership","cosine_similarity":0.17167737713678186,"fuzzy_score":0},{"concept":"George's Farmers Market Chicken Wings, at Sam's Club","cosine_similarity":0.07507578805263351,"fuzzy_score":0},{"concept":"Tyson Products, select varieties, spend $20 at Sam's Club","cosine_similarity":0.06966444274341704,"fuzzy_score":0},{"concept":"Spend $110 on a Full-Priced new Plus Membership and receive an ADDITIONAL 10,000 points","cosine_similarity":0.037681749600967825,"fuzzy_score":0}],
"LOG_LIST":["Input text received was \"sams club membership\"","Input categorized by QueryTypeDetection: 0 with confidence: 0.9963080883026123","Similarity matches with retailers: [{'concept': 'SAMS CLUB', 'cosine_similarity': 0.6725889529793223, 'fuzzy_score': 0}]"]}

### Testing
curl -X POST http://172.174.142.5:5000/get_offers -H "Content-Type: application/json" -d '{"text":"sams club membership"}'

All specifications and context of this service can be found in <a href="https://github.com/federico2001/Transformer_Based_IR/blob/main/README.md">this repository</a>.
