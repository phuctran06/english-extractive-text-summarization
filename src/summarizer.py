from nltk.corpus import stopwords

from preprocessing import preprocess_article
from preprocessing import calculate_summary_length

from textrank import calculate_tfidf
from textrank import calculate_similarity
from textrank import build_graph
from textrank import calculate_textrank


#Select important and diverse sentences
def select_top_sentences(
    sentences,
    scores,
    similarity_matrix,
    num_sentences=3,
    lambda_param=0.7
):
    selected_indices = []

    #Select sentences one by one
    while len(selected_indices) < num_sentences:
        best_index = None
        best_score = float("-inf")

        for index, score in scores.items():
            if index in selected_indices:
                continue

            #Calculate redundancy
            if selected_indices:
                redundancy = max(
                    similarity_matrix[index][selected_index]
                    for selected_index in selected_indices
                )
            else:
                redundancy = 0

            #Calculate MMR score
            mmr_score = (
                lambda_param * score
                - (1 - lambda_param) * redundancy
            )

            if mmr_score > best_score:
                best_score = mmr_score
                best_index = index

        if best_index is None:
            break

        selected_indices.append(best_index)

    #Restore original sentence order
    selected_indices.sort()

    #Get selected sentences
    selected_sentences = []

    for index in selected_indices:
        selected_sentences.append(sentences[index])

    return selected_sentences


#Create summary
def create_summary(
    sentences,
    scores,
    similarity_matrix,
    num_sentences=3,
    lambda_param=0.7
):
    #Select important and diverse sentences
    selected_sentences = select_top_sentences(
        sentences,
        scores,
        similarity_matrix,
        num_sentences=num_sentences,
        lambda_param=lambda_param
    )

    #Combine sentences
    summary = " ".join(selected_sentences)

    return summary


#Summarize article
def summarize_article(
    article,
    ratio=0.15,
    threshold=0.06,
    lambda_param=0.7,
    num_sentences=None
):
    #Get English stopwords
    english_stopwords = set(stopwords.words("english"))

    #Preprocess article
    sentences, processed_sentences = preprocess_article(
        article,
        english_stopwords
    )

    #Calculate summary length
    if num_sentences is None:
        num_sentences = calculate_summary_length(
            len(sentences),
            ratio
        )
    else:
        num_sentences = min(
            num_sentences,
            len(sentences)
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
        threshold=threshold
    )

    #Calculate TextRank
    scores = calculate_textrank(graph)

    #Create summary
    summary = create_summary(
        sentences,
        scores,
        similarity_matrix,
        num_sentences=num_sentences,
        lambda_param=lambda_param
    )

    return summary


if __name__ == "__main__":
    from load_data import load_cnn_dailymail

    #Load dataset
    dataset = load_cnn_dailymail(split="train")

    #Get first article
    article = dataset[0]["article"]

    #Create summary
    summary = summarize_article(
        article,
        ratio=0.15,
        threshold=0.06,
        lambda_param=0.7
    )

    print("\nOriginal article:")
    print(article)

    print("\nGenerated summary:")
    print(summary)

    print("\nReference summary:")
    print(dataset[0]["highlights"])