from textrank import calculate_textrank


def select_top_sentences(sentences, scores, num_sentences=3):
    #Sort sentences by TextRank score
    ranked_sentences = sorted(
        scores.items(),
        key=lambda item: item[1],
        reverse=True
    )

    #Select top sentences
    selected_indices = [
        index
        for index, score in ranked_sentences[:num_sentences]
    ]

    #Restore original sentence order
    selected_indices.sort()

    #Get selected sentences
    selected_sentences = [
        sentences[index]
        for index in selected_indices
    ]

    return selected_sentences


def create_summary(sentences, scores, num_sentences=3):
    #Select important sentences
    selected_sentences = select_top_sentences(
        sentences,
        scores,
        num_sentences
    )

    #Combine sentences into summary
    summary = " ".join(selected_sentences)

    return summary