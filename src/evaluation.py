from rouge_score import rouge_scorer

from load_data import load_cnn_dailymail
from summarizer import summarize_article


#Calculate ROUGE scores
def calculate_rouge(reference, candidate):
    scorer = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"],
        use_stemmer=True
    )

    scores = scorer.score(
        reference,
        candidate
    )

    return scores


#Evaluate dataset
def evaluate_dataset(
    dataset,
    num_samples=100,
    num_sentences=3,
    threshold=0.06
):
    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    #Evaluate each article
    for i in range(num_samples):
        article = dataset[i]["article"]
        reference = dataset[i]["highlights"]

        candidate = summarize_article(
            article,
            num_sentences=num_sentences,
            threshold=threshold
        )

        scores = calculate_rouge(
            reference,
            candidate
        )

        rouge1_scores.append(
            scores["rouge1"].fmeasure
        )

        rouge2_scores.append(
            scores["rouge2"].fmeasure
        )

        rougeL_scores.append(
            scores["rougeL"].fmeasure
        )

    #Calculate average scores
    average_rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    average_rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    average_rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return {
        "ROUGE-1": average_rouge1,
        "ROUGE-2": average_rouge2,
        "ROUGE-L": average_rougeL
    }


#Run sentence count experiment
def run_sentence_experiment(
    dataset,
    sentence_counts,
    num_samples=100,
    threshold=0.06
):
    results = []

    for num_sentences in sentence_counts:
        print(
            f"\nTesting number of sentences: {num_sentences}"
        )

        scores = evaluate_dataset(
            dataset,
            num_samples=num_samples,
            num_sentences=num_sentences,
            threshold=threshold
        )

        results.append({
            "num_sentences": num_sentences,
            "ROUGE-1": scores["ROUGE-1"],
            "ROUGE-2": scores["ROUGE-2"],
            "ROUGE-L": scores["ROUGE-L"]
        })

        print(
            f"ROUGE-1: {scores['ROUGE-1']:.4f}"
        )

        print(
            f"ROUGE-2: {scores['ROUGE-2']:.4f}"
        )

        print(
            f"ROUGE-L: {scores['ROUGE-L']:.4f}"
        )

    return results


if __name__ == "__main__":
    #Load test dataset
    dataset = load_cnn_dailymail(split="test")

    #Define sentence counts
    sentence_counts = [
        2,
        3,
        4,
        5
    ]

    #Run experiment
    results = run_sentence_experiment(
        dataset,
        sentence_counts,
        num_samples=100,
        threshold=0.06
    )

    print("\nFinal results:")

    for result in results:
        print(
            f"Sentences: {result['num_sentences']} | "
            f"ROUGE-1: {result['ROUGE-1']:.4f} | "
            f"ROUGE-2: {result['ROUGE-2']:.4f} | "
            f"ROUGE-L: {result['ROUGE-L']:.4f}"
        )