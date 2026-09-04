from datasets import load_dataset

#Load dataset from Hugging Face
def load_cnn_dailymail(split="train"):
    dataset = load_dataset("abisee/cnn_dailymail","3.0.0",split=split)

    return dataset


if __name__ == "__main__":
    dataset = load_cnn_dailymail()

    print(dataset)
    print("\nNumber of articles:", len(dataset))

    print("\nFirst article:")
    print(dataset[0]["article"])

    print("\nReference summary:")
    print(dataset[0]["highlights"])