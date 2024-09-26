import pandas as pd
import nltk
import string
from nltk.corpus import stopwords
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
import openai
import os
import time  # For measuring time
# Optional for resource monitoring
import psutil

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

# Initialize SBERT model
sbert_model = SentenceTransformer('all-mpnet-base-v2')


# Preprocessing function for text (common across all methods)
def preprocess(text):
    text = text.translate(str.maketrans('', '', string.punctuation))
    text = text.lower()
    stop_words = set(stopwords.words('english'))
    return ' '.join(word for word in text.split() if word not in stop_words)


# OpenAI embeddings function
def get_openai_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response.data[0].embedding


# LLM similarity computation
def compute_similarity_llm(query, documents):
    query = preprocess(query)
    query_embedding = get_openai_embedding(query)
    document_embeddings = [get_openai_embedding(preprocess(doc['description'])) for doc in documents]
    similarities = cosine_similarity([query_embedding], document_embeddings)[0]
    return similarities


# SBERT similarity computation
def compute_similarity_sbert(query, documents):
    query = preprocess(query)
    query_embedding = sbert_model.encode(query, convert_to_tensor=True)
    document_embeddings = sbert_model.encode([preprocess(doc['description']) for doc in documents],
                                             convert_to_tensor=True)
    cosine_similarities = util.pytorch_cos_sim(query_embedding, document_embeddings)
    return cosine_similarities[0].tolist()


# Word2Vec (TF-IDF) similarity computation
def compute_similarity_word2vec(query, documents):
    vectorizer = TfidfVectorizer()
    docs = [preprocess(doc['description']) for doc in documents]
    query = preprocess(query)
    docs.append(query)
    tfidf_matrix = vectorizer.fit_transform(docs)
    cosine_sim = cosine_similarity(tfidf_matrix[-1], tfidf_matrix[:-1])
    return cosine_sim.flatten()


# Function to classify a single summary
def classify_summary(summary, method):
    if method == 'llm':
        similarities = compute_similarity_llm(summary, data)
    elif method == 'sbert':
        similarities = compute_similarity_sbert(summary, data)
    else:  # word2vec
        similarities = compute_similarity_word2vec(summary, data)

    # Sort similarities and pick top 3 terms
    indexed_similarities = list(enumerate(similarities))
    sorted_similarities = sorted(indexed_similarities, key=lambda x: x[1], reverse=True)[:3]
    top_terms = [data[i]['term'] for i, score in sorted_similarities if score > 0]

    # Return classifications or N/A
    if not top_terms:
        return 'N/A'
    return ', '.join(top_terms[:3])  # Limit to top 3 classifications


# Function to measure time for classification and display KPIs
def measure_kpi(df, method):
    total_time = 0
    summaries_processed = 0

    for index, row in df.iterrows():
        summary = row['Summary']
        start_time = time.time()

        # Classify the summary using the given method
        classify_summary(summary, method)

        # Measure time taken for this summary
        time_taken = time.time() - start_time
        total_time += time_taken
        summaries_processed += 1

    # Calculate average time per summary
    average_time = total_time / summaries_processed if summaries_processed > 0 else 0

    # Optionally gather system resource usage (CPU, memory)
    cpu_usage = psutil.cpu_percent(interval=1)
    memory_info = psutil.virtual_memory()

    # Print KPI metrics
    print(f"{method.upper()} :")
    print(f"Documents/Summaries processed: {summaries_processed}")
    print(f"Average time to classify a single document/summary: {average_time:.4f} seconds")
    print(f"CPU Usage: {cpu_usage}%")
    print(f"Memory Usage: {memory_info.percent}%\n")


# Process CSV file and generate results
def process_csv(input_file):
    # Load input CSV
    df = pd.read_csv(input_file)

    # Create output directory if not exists
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)

    # Initialize result DataFrames
    df_llm = df.copy()
    df_sbert = df.copy()
    df_word2vec = df.copy()

    # Measure and display KPIs for each method
    measure_kpi(df, 'llm')
    measure_kpi(df, 'sbert')
    measure_kpi(df, 'word2vec')

    # Iterate over each row in the CSV file and apply classification methods
    for index, row in df.iterrows():
        summary = row['Summary']

        # Apply each method
        df_llm.at[index, 'classification'] = classify_summary(summary, 'llm')
        df_sbert.at[index, 'classification'] = classify_summary(summary, 'sbert')
        df_word2vec.at[index, 'classification'] = classify_summary(summary, 'word2vec')

    # Save results to CSV files
    df_llm.to_csv(os.path.join(output_dir, 'llm_result.csv'), index=False)
    df_sbert.to_csv(os.path.join(output_dir, 'sbert_result.csv'), index=False)
    df_word2vec.to_csv(os.path.join(output_dir, 'word2vec_result.csv'), index=False)


# Run the process on the CSV file
if __name__ == '__main__':
    input_csv = 'dataset.csv'
    process_csv(input_csv)