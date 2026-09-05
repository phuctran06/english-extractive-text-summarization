import re
import nltk


#Split article into sentences
def split_sentences(text):
    sentences = nltk.sent_tokenize(text)

    return sentences


#Clean sentence
def clean_sentence(sentence):
    #Remove extra whitespace
    sentence = re.sub(r"\s+", " ", sentence)

    #Remove leading and trailing whitespace
    sentence = sentence.strip()

    return sentence


#Preprocess sentence
def preprocess_sentence(sentence, stopwords):
    #Convert to lowercase
    sentence = sentence.lower()

    #Remove punctuation
    sentence = re.sub(r"[^a-zA-Z0-9\s]", "", sentence)

    #Split sentence into words
    words = sentence.split()

    #Remove stopwords
    words = [
        word
        for word in words
        if word not in stopwords
    ]

    return words


#Preprocess article
def preprocess_article(article, stopwords):
    #Split article into sentences
    sentences = split_sentences(article)

    #Clean sentences
    cleaned_sentences = [
        clean_sentence(sentence)
        for sentence in sentences
        if sentence.strip()
    ]

    #Create processed sentences
    processed_sentences = [
        preprocess_sentence(sentence, stopwords)
        for sentence in cleaned_sentences
    ]

    return cleaned_sentences, processed_sentences


#Calculate summary sentence count
def calculate_summary_length(num_sentences, ratio):
    #Calculate target number of sentences
    target_sentences = round(
        num_sentences * ratio
    )

    #Keep at least one sentence
    target_sentences = max(
        1,
        target_sentences
    )

    #Do not exceed article length
    target_sentences = min(
        num_sentences,
        target_sentences
    )

    return target_sentences


if __name__ == "__main__":
    from nltk.corpus import stopwords
    from load_data import load_cnn_dailymail

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

    print("Number of sentences:", len(sentences))

    print("\nSentences:")

    for i, sentence in enumerate(sentences):
        print(f"{i + 1}: {sentence}")

    print("\nOriginal:")
    print(sentences[0])

    print("\nProcessed:")
    print(processed_sentences[0])

    print(
        "\n15% summary:",
        calculate_summary_length(
            len(sentences),
            0.15
        ),
        "sentences"
    )