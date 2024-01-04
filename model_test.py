from transformers import LongformerTokenizer, LongformerForSequenceClassification
import torch
import torch.nn.functional as F

def analyze_sentiment_longformer(reviews):
    # Load the Longformer tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

    # Tokenize the reviews
    tokenized_reviews = tokenizer(reviews, return_tensors='pt', padding=True, truncation=True)

    # Make predictions using the Longformer model
    with torch.no_grad():
        outputs = model(**tokenized_reviews)

    # Extract the predicted logits
    logits = outputs.logits

    # Apply softmax to get probabilities
    probabilities = F.softmax(logits, dim=1)

    # Determine the predicted sentiment labels
    predicted_labels = torch.argmax(probabilities, dim=1)

    # Convert tensor to list
    predicted_labels = predicted_labels.tolist()

    # Map labels to 'POSITIVE' or 'NEGATIVE'
    sentiment_labels = ['POSITIVE' if label == 1 else 'NEGATIVE' for label in predicted_labels]

    # Create results dictionary
    sentiment_results = [{'label': label, 'score': prob.item()} for label, prob in zip(sentiment_labels, probabilities[:, 1])]

    return sentiment_results

# Example usage:
reviews = ["I love this product!", "This is a terrible company. I woudn't recommend their products to anyone!"]
sentiment_results_longformer = analyze_sentiment_longformer(reviews)

#print sentiment_results_longformer
print(sentiment_results_longformer)
