import requests
from bs4 import BeautifulSoup
from transformers import pipeline

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

def scrape_reviews(url):
    reviews = []
    for page in range(1, 12):  # Assuming there are 11 pages of reviews
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

# Function to perform sentiment analysis
def analyze_sentiment(reviews):
    sentiment_analyzer = pipeline('sentiment-analysis')
    results = sentiment_analyzer(reviews)
    return results

# Function to display results
def display_results(sentiment_results):
    average_sentiment = sum([result['score'] for result in sentiment_results]) / len(sentiment_results)
    print(f"Average Sentiment Score: {average_sentiment:.2f}")

    sorted_results = sorted(sentiment_results, key=lambda x: x['score'], reverse=True)
    print("\nTop 10 Positive Reviews:")
    for i in range(10):
        print(f"{i+1}. {sorted_results[i]['label']} - {sorted_results[i]['score']:.4f}")

    print("\nTop 10 Negative Reviews:")
    for i in range(-1, -11, -1):
        print(f"{-i}. {sorted_results[i]['label']} - {sorted_results[i]['score']:.4f}")

# Main execution
if __name__ == "__main__":
    trustpilot_url = "https://www.trustpilot.com/review/sunlife.ca"
    reviews = scrape_reviews2(trustpilot_url)
    # print the reviews length
    print(len(reviews))

    #sentiment_results = analyze_sentiment(reviews)
    #display_results(sentiment_results)
