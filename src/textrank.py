from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

import networkx as nx

from preprocessing import preprocess_article
from load_data import load_cnn_dailymail

from nltk.corpus import stopwords


#Calculate TF-IDF
def calculate_tfidf(processed_sentences):
    #Convert list of words into strings
    processed_sentences = [
        " ".join(words)
        for words in processed_sentences
    ]

    #Create TF-IDF vectorizer
    vectorizer = TfidfVectorizer()

    #Convert sentences into TF-IDF vectors
    tfidf_matrix = vectorizer.fit_transform(
        processed_sentences
    )

    return tfidf_matrix, vectorizer


#Calculate cosine similarity
def calculate_similarity(tfidf_matrix):
    #Calculate similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


#Build sentence graph
def build_graph(similarity_matrix, threshold=0.06):
    #Create empty graph
    graph = nx.Graph()

    #Add sentence nodes
    for i in range(similarity_matrix.shape[0]):
        graph.add_node(i)

    #Add edges between similar sentences
    for i in range(similarity_matrix.shape[0]):
        for j in range(i + 1, similarity_matrix.shape[0]):
            similarity = similarity_matrix[i][j]

            if similarity >= threshold:
                graph.add_edge(
                    i,
                    j,
                    weight=similarity
                )

    return graph


#Calculate TextRank
def calculate_textrank(graph):
    #Calculate TextRank scores
    scores = nx.pagerank(
        graph,
        weight="weight"
    )

    return scores


if __name__ == "__main__":
    #Load dataset
    dataset = load_cnn_dailymail(split="train")

    #Get first article
    article = dataset[0]["article"]

    #Get English stopwords
    english_stopwords = set(stopwords.words("english"))

    #Preprocess article
    sentences, processed_sentences = preprocess_article(
        article,
        english_stopwords
    )

    #Calculate TF-IDF
    tfidf_matrix, vectorizer = calculate_tfidf(
        processed_sentences
    )

    #Calculate similarity
    similarity_matrix = calculate_similarity(
        tfidf_matrix
    )

    #Build graph
    graph = build_graph(
        similarity_matrix,
        threshold=0.06
    )

    #Calculate TextRank
    scores = calculate_textrank(graph)

    print("Number of sentences:", len(sentences))
    print("Number of words:", len(vectorizer.get_feature_names_out()))
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("Similarity shape:", similarity_matrix.shape)

    print("\nNumber of nodes:", graph.number_of_nodes())
    print("Number of edges:", graph.number_of_edges())

    print("\nTextRank scores:")

    for node, score in scores.items():
        print(
            f"Sentence {node + 1}: {score:.4f}"
        )