import re
import nltk


#Split article into sentences
def split_sentences(text):
    sentences = nltk.sent_tokenize(text)

    return sentences

def clean_sentence(sentence):
    #Remove extra whitespace
    sentence = re.sub(r"\s+", " ", sentence)

    #Remove leading and trailing whitespace
    sentence = sentence.strip()

    return sentence


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
