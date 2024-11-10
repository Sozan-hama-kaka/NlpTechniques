from flask import Flask, request, jsonify
import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
from gensim.models import KeyedVectors
import numpy as np
from dotenv import load_dotenv
import os

# Download NLTK stopwords if not already installed
nltk.download('stopwords')

app = Flask(__name__)

load_dotenv()

openai.api_key = os.getenv('OPENAI_KEY', 'NULL')

# Load pre-trained Word2Vec model (Google News vectors)
w2v_model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)

data = [
    {
        'term': 'Artificial Intelligence',
        'description': 'Covers all areas of AI except Vision, Robotics, Machine Learning, Multiagent Systems, and Computation and Language (Natural Language Processing), which have separate subject areas. In particular, includes Expert Systems, Theorem Proving (although this may overlap with Logic in Computer Science), Knowledge Representation, Planning, and Uncertainty in AI.'
    },
    {
        'term': 'Computation and Language',
        'description': 'Covers natural language processing. Roughly includes material in ACM Subject Class I.2.7. Note that work on artificial languages (programming languages, logics, formal systems) that does not explicitly address natural-language issues broadly construed (natural-language processing, computational linguistics, speech, text retrieval, etc.) is not appropriate for this area.'
    },
    {
        'term': 'Computer Vision and Pattern Recognition',
        'description': 'Covers image processing, computer vision, pattern recognition, and scene understanding. Roughly includes material in ACM Subject Classes I.2.10, I.4, and I.5.'
    },
    {
        'term': 'Databases',
        'description': 'Covers database management, datamining, and data processing.'
    },
    {
        'term': 'Data Structures and Algorithms',
        'description': 'Covers data structures and analysis of algorithms.'
    },
    {
        'term': 'Human-Computer Interaction',
        'description': 'Covers human factors, user interfaces, and collaborative computing. Roughly includes material in ACM Subject Classes H.1.2 and all of H.5, except for H.5.1, which is more likely to have Multimedia as the primary subject area.'
    },
    {
        'term': 'Machine Learning',
        'description': 'Papers on all aspects of machine learning research (supervised, unsupervised, reinforcement learning, bandit problems, and so on) including also robustness, explanation, fairness, and methodology. cs.LG is also an appropriate primary category for applications of machine learning methods.'
    },
    {
        'term': 'Multiagent Systems',
        'description': 'Covers multiagent systems, distributed artificial intelligence, intelligent agents, coordinated interactions. and practical applications.'
    },
    {
        'term': 'Networking and Internet Architecture',
        'description': 'Covers all aspects of computer communication networks, including network architecture and design, network protocols, and internetwork standards (like TCP/IP). Also includes topics, such as web caching, that are directly relevant to Internet architecture and performance.'
    },
    {
        'term': 'Software Engineering',
        'description': 'Covers design tools, software metrics, testing and debugging, programming environments, etc. Roughly includes material in all of ACM Subject Classes D.2, except that D.2.4 (program verification) should probably have Logics in Computer Science as the primary subject area.'
    }
]


# Preprocessing function common to all methods
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    text = ' '.join(word for word in text.split() if word not in stop_words)
    return text


def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


def compute_similarity_llm(query, documents):
    print("Processing with LLM...")
    query = preprocess(query)
    descriptions = [preprocess(doc['description']) for doc in documents]
    query_embedding = get_openai_embedding(query)
    document_embeddings = [get_openai_embedding(description) for description in descriptions]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    return {"method": "Result processed with LLM", "similarities": similarities.tolist()}


# SBERT-based similarity
model = SentenceTransformer('all-mpnet-base-v2')


def compute_similarity_sbert(query, documents):
    print("Processing with SBERT...")
    query = preprocess(query)
    descriptions = [preprocess(doc['description']) for doc in documents]
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(descriptions, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    return {"method": "Result processed with SBERT", "similarities": cosine_similarities[0].tolist()}


# Word2Vec-based similarity (TF-IDF)
def compute_similarity_word2vec(query, documents):
    print("Processing with Word2Vec...")
    vectorizer = TfidfVectorizer()
    docs = [preprocess(doc['description']) for doc in documents]
    query = preprocess(query)
    docs.append(query)
    tfidf_matrix = vectorizer.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return {"method": "Result processed with Word2Vec", "similarities": cosine_sim.flatten()}


# Word Mover's Distance (WMD)
def compute_wmd_similarity(query, documents):
    print("Processing with WMD...")
    query = preprocess(query).split()
    similarities = []
    for doc in documents:
        doc_preprocessed = preprocess(doc['description']).split()
        distance = w2v_model.wmdistance(query, doc_preprocessed)
        similarities.append(1 / (1 + distance))  # Convert distance to similarity
    return {"method": "Result processed with WMD", "similarities": similarities}


# API route
@app.route('/compare', methods=['POST'])
def compare_documents():
    summary = request.args.get('summary')
    method = request.args.get('method', '').lower()
    measurement = 'cosine'  # Default measurement

    if not summary:
        return jsonify({'error': 'No summary provided'}), 400
    if method not in ['llm', 'sbert', 'word2vec']:
        return jsonify({'error': 'Invalid method provided. Choose "llm", "sbert", or "word2vec".'}), 400

    # Compute similarities based on the selected method
    if method == 'llm':
        result = compute_similarity_llm(summary, data)
    elif method == 'sbert':
        result = compute_similarity_sbert(summary, data)
    else:
        result = compute_similarity_word2vec(summary, data)

    # Apply chosen measurement method
    similarities = result["similarities"]
    indexed_similarities = list(enumerate(similarities))
    sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)

    # Check for semantic novelty
    novelty_threshold = 0.8  # Adjust this threshold as needed
    top_score = sorted_similarities[0][1]
    if top_score < novelty_threshold:
        return jsonify({
            "method": result["method"],
            "message": "Semantic Novelty detected in Document Abstract! The provided document abstract / summary cannot be classified into existing classifications."
        })

    # Get the top 3 matches if novelty is not detected
    top_matches = sorted_similarities[:3]
    response_text = [
        {
            "term": data[i]['term'],
            "description": data[i]['description'],
            "similarity_score": score,
            "summary": summary
        }
        for i, score in top_matches
    ]

    response_text.insert(0, {"method": result["method"]})

    return jsonify(response_text)


if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)