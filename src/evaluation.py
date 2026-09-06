from datasets import load_dataset
from rouge_score import rouge_scorer

from summarizer import summarize_article


#Evaluate summarization performance
def evaluate_dataset(split="test", num_samples=100, ratio=0.15, threshold=0.06, lambda_param=0.7):
    #Load dataset
    dataset = load_dataset("abisee/cnn_dailymail", "3.0.0", split=split)

    #Limit number of samples
    dataset = dataset.select(range(min(num_samples, len(dataset))))

    #Create ROUGE scorer
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    rouge1_scores = []
    rouge2_scores = []
    rougeL_scores = []

    #Evaluate articles
    for i, item in enumerate(dataset):
        article = item["article"]
        reference = item["highlights"]

        summary = summarize_article(article, ratio=ratio, threshold=threshold, lambda_param=lambda_param)
        scores = scorer.score(reference, summary)

        rouge1_scores.append(scores["rouge1"].fmeasure)
        rouge2_scores.append(scores["rouge2"].fmeasure)
        rougeL_scores.append(scores["rougeL"].fmeasure)

        print(f"Article {i + 1}/{len(dataset)} completed")

    #Calculate average scores
    rouge1 = sum(rouge1_scores) / len(rouge1_scores)
    rouge2 = sum(rouge2_scores) / len(rouge2_scores)
    rougeL = sum(rougeL_scores) / len(rougeL_scores)

    return rouge1, rouge2, rougeL


if __name__ == "__main__":
    #Evaluate model
    rouge1, rouge2, rougeL = evaluate_dataset(
        split="test",
        num_samples=100,
        ratio=0.15,
        threshold=0.06,
        lambda_param=0.7
    )

    print("\nEvaluation Results")
    print("------------------")
    print(f"ROUGE-1: {rouge1:.4f}")
    print(f"ROUGE-2: {rouge2:.4f}")
    print(f"ROUGE-L: {rougeL:.4f}")