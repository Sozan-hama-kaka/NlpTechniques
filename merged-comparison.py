import nltk
from nltk.corpus import stopwords
import string
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import openai

# Download NLTK stopwords if not already installed
nltk.download('stopwords')

openai.api_key = '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'

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


# OpenAI LLM embedding
def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"  # Example OpenAI embedding model
    )
    return response.data[0].embedding


# Compute similarity using LLM
def compute_similarity_llm(query, documents):
    print("Processing with LLM...")
    query = preprocess(query)
    descriptions = [preprocess(doc['description']) for doc in documents]
    query_embedding = get_openai_embedding(query)
    document_embeddings = [get_openai_embedding(description) for description in descriptions]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    return {"method": "LLM", "similarities": similarities.tolist()}


# SBERT model initialization
model = SentenceTransformer('all-mpnet-base-v2')


# Compute similarity using SBERT
def compute_similarity_sbert(query, documents):
    print("Processing with SBERT...")
    query = preprocess(query)
    descriptions = [preprocess(doc['description']) for doc in documents]
    query_embedding = model.encode(query, convert_to_tensor=True)
    document_embeddings = model.encode(descriptions, convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    return {"method": "SBERT", "similarities": cosine_similarities[0].tolist()}


# Compute similarity using Word2Vec (TF-IDF)
def compute_similarity_word2vec(query, documents):
    print("Processing with Word2Vec...")
    vectorizer = TfidfVectorizer()
    docs = [preprocess(doc['description']) for doc in documents]
    query = preprocess(query)
    docs.append(query)
    tfidf_matrix = vectorizer.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return {"method": "Word2Vec", "similarities": cosine_sim.flatten()}


# Sort and return top 3 matches
def get_top_matches(similarities, documents, summary):
    indexed_similarities = list(enumerate(similarities))
    sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)
    top_matches = sorted_similarities[:3]

    response_text = [
        {
            "term": documents[i]['term'],
            "description": documents[i]['description'],
            "similarity_score": score,
            "summary": summary
        }
        for i, score in top_matches
    ]
    return response_text


# Main function to process the input from the command line interactively
def main():
    # Prompt the user for input
    summary = input("Enter your text to analyze: ")

    # Run all methods and get top 3 matches
    results = []

    llm_result = compute_similarity_llm(summary, data)
    results.append(
        {"method": llm_result["method"], "matches": get_top_matches(llm_result["similarities"], data, summary)})

    sbert_result = compute_similarity_sbert(summary, data)
    results.append(
        {"method": sbert_result["method"], "matches": get_top_matches(sbert_result["similarities"], data, summary)})

    word2vec_result = compute_similarity_word2vec(summary, data)
    results.append({"method": word2vec_result["method"],
                    "matches": get_top_matches(word2vec_result["similarities"], data, summary)})

    # Output the results for each method
    for result in results:
        print(f"Results for {result['method']}:")
        for match in result['matches']:
            print(f" - Term: {match['term']}, Similarity: {match['similarity_score']:.4f}")
        print()


if __name__ == '__main__':
    main()
