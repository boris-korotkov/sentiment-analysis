import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import BartTokenizer, BartForSequenceClassification




# Function to scrape reviews from Trustpilot

# Get number of pages
def get_total_pages(url):
    response = requests.get(url)
    if response.status_code == 200:
        soup = BeautifulSoup(response.text, 'html.parser')
        # Assuming the pagination button has the attribute name="pagination-button-last"
        last_page_element = soup.find('a', attrs={'name': 'pagination-button-last'})
        if last_page_element:
            return int(last_page_element.text.strip())
    return 0

# Get reviews by page number
def scrape_reviews(url):
    reviews = []
    total_pages = get_total_pages(url)
    
    for page in range(1, total_pages + 1):
        page_url = f"{url}?page={page}"
        response = requests.get(page_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Update the class names based on the current HTML structure
            review_elements = soup.find_all('p', class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
            
            # Check if there are any reviews on the page
            if review_elements:
                reviews.extend([review.text.strip() for review in review_elements])
            else:
                # If no reviews are found on the page, break the loop
                break
        else:
            # If the request is unsuccessful, print an error message
            print(f"Failed to fetch reviews from {page_url}. Status code: {response.status_code}")
            break

    return reviews

# Function to perform sentiment analysis

def analyze_sentiment(reviews):
    sentiment_analyzer = pipeline('sentiment-analysis')
    results = sentiment_analyzer(reviews)
    return results


def analyze_sentiment_longformer(reviews):
    # Load the tokenizer and model
    tokenizer = BartTokenizer.from_pretrained("facebook/bart-large-cnn")
    model = BartForSequenceClassification.from_pretrained("facebook/bart-large-cnn")



    # Create a sentiment analysis pipeline using the  model
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    print(sentiment_analyzer.model.config)

    
    results = sentiment_analyzer(reviews)
    return results


# Function to display results

def display_results(sentiment_results, reviews):
    
    # Combine sentiment results with their corresponding reviews
    results_with_reviews = list(zip(sentiment_results, reviews))

    # Sort results based on scores
    sorted_results = sorted(results_with_reviews, key=lambda x: x[0]['score'], reverse=True)

    # Display average sentiment score
    average_sentiment = sum([result[0]['score'] for result in results_with_reviews]) / len(results_with_reviews)
    print(f"Average Sentiment Score: {average_sentiment:.4f}")

    # Display top 5 positive reviews with content and score
    print("\nTop 5 Positive Reviews:")
    for i in range(min(5, len(sorted_results))):
        print(f"{i+1}. {sorted_results[i][1]} - Score: {sorted_results[i][0]['score']:.4f}")

    # Display top 5 negative reviews with content and score
    print("\nTop 5 Negative Reviews:")
    for i in range(-1, -min(6, len(sorted_results)), -1):
        print(f"{-i}. {sorted_results[i][1]} - Score: {sorted_results[i][0]['score']:.4f}")
        

# Main execution
if __name__ == "__main__":
    trustpilot_url = "https://www.trustpilot.com/review/sunlife.ca"
    reviews = scrape_reviews(trustpilot_url)
    
    # print the reviews length
    print(len(reviews))

    #sentiment_results = analyze_sentiment(reviews)
    sentiment_results = analyze_sentiment_longformer(reviews)
    display_results(sentiment_results,reviews)
    #display_results_old(sentiment_results)