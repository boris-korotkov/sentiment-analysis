import requests
from bs4 import BeautifulSoup
from transformers import pipeline
from transformers import LongformerTokenizer, LongformerForSequenceClassification

# Function to scrape reviews from Trustpilot

# the number of total pages is stored in the A tag with the element name pagination-button-last. I need to extract the number of pages from this element and use it in the for loop to scrape all the pages.
def scrape_reviews2(url):
    reviews = []
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    # get the number of pages
    pagination_button_last = soup.find('a', class_='pagination-button pagination-button--last')
    # get the href attribute
    last_page_url = pagination_button_last.get('href')
    # get the number of pages from the last page url
    last_page_number = int(last_page_url.split('=')[1])
    # loop through all the pages
    for page in range(1, last_page_number + 1):
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
    return reviews

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
    # Load the Longformer tokenizer and model
    tokenizer = LongformerTokenizer.from_pretrained("allenai/longformer-base-4096")
    model = LongformerForSequenceClassification.from_pretrained("allenai/longformer-base-4096")

    # Create a sentiment analysis pipeline using the Longformer model
    sentiment_analyzer = pipeline('sentiment-analysis', model=model, tokenizer=tokenizer)
    print(sentiment_analyzer.model.config)

    
    results = sentiment_analyzer(reviews)
    return results


# Function to display results
def display_results_old(sentiment_results):
    average_sentiment = sum([result['score'] for result in sentiment_results]) / len(sentiment_results)
    print(f"Average Sentiment Score: {average_sentiment:.2f}")

    sorted_results = sorted(sentiment_results, key=lambda x: x['score'], reverse=True)
    print("\nTop 10 Positive Reviews:")
    for i in range(10):
        print(f"{i+1}. {sorted_results[i]['label']} - {sorted_results[i]['score']:.4f}")

    print("\nTop 10 Negative Reviews:")
    for i in range(-1, -11, -1):
        print(f"{-i}. {sorted_results[i]['label']} - {sorted_results[i]['score']:.4f}")

def display_results(sentiment_results, reviews):
    # Combine sentiment results with their corresponding reviews
    results_with_reviews = list(zip(sentiment_results, reviews))

    # Sort results based on scores
    sorted_results = sorted(results_with_reviews, key=lambda x: x[0]['score'], reverse=True)

    # Display average sentiment score
    average_sentiment = sum([result[0]['score'] for result in sentiment_results]) / len(sentiment_results)
    print(f"Average Sentiment Score: {average_sentiment:.4f}")

    # Display top 10 positive reviews with content and score
    print("\nTop 5 Positive Reviews:")
    for i in range(min(5, len(sorted_results))):
        print(f"{i+1}. {sorted_results[i][1]} - Score: {sorted_results[i][0]['score']:.4f}")

    # Display top 5 negative reviews with content and score
    print("\nTop 5 Negative Reviews:")
    for i in range(-1, -6, -1):
        print(f"{-i}. {sorted_results[i][1]} - Score: {sorted_results[i][0]['score']:.4f}")
        

# Main execution
if __name__ == "__main__":
    trustpilot_url = "https://www.trustpilot.com/review/sunlife.ca"
    reviews = scrape_reviews(trustpilot_url)
    
    # print the reviews length
    print(len(reviews))

    #sentiment_results = analyze_sentiment(reviews)
    sentiment_results = analyze_sentiment_longformer(reviews)
    display_results(sentiment_results)
