from transformers import LongformerTokenizer, LongformerForSequenceClassification
from transformers import pipeline

# Load the Longformer tokenizer and model
tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

# Create a sentiment analysis pipeline using the Longformer model
sentiment_pipeline = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)

# Example usage:
result = sentiment_pipeline("I love using the Longformer model for sentiment analysis!")
print(result)
