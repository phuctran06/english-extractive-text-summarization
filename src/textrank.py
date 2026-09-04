from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from preprocessing import preprocess_article
from load_data import load_cnn_dailymail

from nltk.corpus import stopwords


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


def calculate_similarity(tfidf_matrix):
    #Calculate cosine similarity between sentences
    similarity_matrix = cosine_similarity(tfidf_matrix)

    return similarity_matrix


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

    #Calculate cosine similarity
    similarity_matrix = calculate_similarity(
        tfidf_matrix
    )

    print("Number of sentences:", len(sentences))
    print("Number of words:", len(vectorizer.get_feature_names_out()))
    print("TF-IDF shape:", tfidf_matrix.shape)
    print("Similarity shape:", similarity_matrix.shape)

    print("\nSimilarity matrix:")
    print(similarity_matrix)