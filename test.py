from datasets import load_dataset

dataset = load_dataset("abisee/cnn_dailymail", "3.0.0")

print(dataset)

article = dataset["train"][0]["article"]
summary = dataset["train"][0]["highlights"]

print("ARTICLE:")
print(article)

print("\nREFERENCE SUMMARY:")
print(summary)