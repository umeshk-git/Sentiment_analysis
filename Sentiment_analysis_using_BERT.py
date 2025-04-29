import torch
from transformers import BertTokenizer, BertForSequenceClassification
import sys


def predict_sentiment(text):
    # loading pre-trained model and tokenizer
    model_name = "nlptown/bert-base-multilingual-uncased-sentiment"
    tokenizer = BertTokenizer.from_pretrained(model_name)
    model = BertForSequenceClassification.from_pretrained(model_name)

    # tokenize the input
    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)

    # run model
    with torch.no_grad():
        outputs = model(**inputs)
    
    logits = outputs.logits
    predicted_class = torch.argmax(logits, dim=1).item()

    # map the value to rating
    label_map = {
        0: "1 star (Very Negative)",
        1: "2 stars (Negative)",
        2: "3 stars (Neutral)",
        3: "4 stars (Positive)",
        4: "5 stars (Very Positive)"
    }

    return label_map[predicted_class]


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Please provide a sentence for sentiment prediction")
        print("Usage: Python sentiment_analysis_using_BERT.py 'Your sentence here'")
        sys.exit()
    
    input_text = " ".join(sys.argv[1:])
    sentiment = predict_sentiment(input_text)
    print(f" Input: {input_text}")
    print(f" Predicted sentiment: {sentiment}")
