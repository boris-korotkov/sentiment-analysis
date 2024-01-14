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
    reviews_with_titles  = []
    total_pages = get_total_pages(url)
    
    for page in range(1, total_pages + 1):
        page_url = f"{url}?page={page}"
        response = requests.get(page_url)
        
        if response.status_code == 200:
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Update the class names based on the current HTML structure
            review_elements = soup.find_all('p', class_='typography_body-l__KUYFJ typography_appearance-default__AAY17 typography_color-black__5LYEn')
            title_elements = soup.find_all('h2', class_='typography_heading-s__f7029 typography_appearance-default__AAY17')
            
            # Check if there are any reviews on the page
            if review_elements and title_elements:
                reviews_with_titles.extend([(title.text.strip(), review.text.strip()) for title, review in zip(title_elements, review_elements)])
            else:
                # If no reviews or titles are found on the page, break the loop
                break

            
           
        else:
            # If the request is unsuccessful, print an error message
            print(f"Failed to fetch reviews from {page_url}. Status code: {response.status_code}")
            break

    return reviews_with_titles

# Function to perform sentiment analysis


def analyze_sentiment(reviews):
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

    # Display top 5 negative reviews with content and score in the desending order of scores by absolute score value
    
    print("\nTop 5 Negative Reviews:")
    #for i in range(-1, -min(6, len(sorted_results)), -1):
    for i in range(len(sorted_results) - 1, len(sorted_results) - min(6, len(sorted_results)) - 1, -1):
        print(f"{-i}. {sorted_results[i][1]} - Score: {sorted_results[i][0]['score']:.4f}")
        
# Function to save sentiment analysis results as an HTML file
def create_html_file(results_with_titles):

    # Separate positive and negative reviews
    positive_reviews = [result for result in results_with_titles if result[0]['label'] == 'LABEL_2']
    negative_reviews = [result for result in results_with_titles if result[0]['label'] == 'LABEL_0']

    # Sort positive reviews based on scores
    sorted_positive_reviews = sorted(positive_reviews, key=lambda x: x[0]['score'], reverse=True)

    # Sort negative reviews based on the absolute value of scores
    sorted_negative_reviews = sorted(negative_reviews, key=lambda x: abs(x[0]['score']), reverse=True)

    # Sort results based on scores
    # sorted_results = sorted(results_with_titles, key=lambda x: x[0]['score'], reverse=True)

    html_content = "<html><head><title>Sentiment Analysis Results</title></head><body>"

    # Display average sentiment score
    average_sentiment = sum([result[0]['score'] for result in results_with_titles]) / len(results_with_titles)
    html_content += f"<h2>Average Sentiment Score: {average_sentiment:.4f}</h2>"

    # Display top 5 positive reviews with titles, content, and score
    html_content += "<h3>Top 5 Positive Reviews:</h3><ol>"
    for i in range(min(5, len(sorted_positive_reviews))):
        title, review = sorted_positive_reviews[i][1]
        html_content += f"<li><strong>{title} - Score: {sorted_positive_reviews[i][0]['score']:.4f}</strong><br/>{review}</li>"
    html_content += "</ol>"

    # Display top 5 negative reviews with titles, content, and absolute score
    html_content += "<h3>Top 5 Negative Reviews:</h3><ol>"
    for i in range(min(5, len(sorted_negative_reviews))):
        title, review = sorted_negative_reviews[i][1]
        html_content += f"<li><strong>{title} - Absolute Score: {abs(sorted_negative_reviews[i][0]['score']):.4f}</strong><br/>{review}</li>"
    html_content += "</ol>"

    html_content += "</body></html>"

    return html_content

def save_html_file(html_content, file_path="sentiment_analysis_results.html"):
    with open(file_path, "w", encoding="utf-8") as html_file:
        html_file.write(html_content)
    print(f"HTML file saved at: {file_path}")

# Main execution
if __name__ == "__main__":
    trustpilot_url = "https://www.trustpilot.com/review/sunlife.ca"
    reviews_with_titles = scrape_reviews(trustpilot_url)
    
        # Perform sentiment analysis
    sentiment_results = analyze_sentiment([review[1] for review in reviews_with_titles])

    # Display results
    # display_results(sentiment_results, [review[1] for review in reviews_with_titles])

    # Combine sentiment results with titles and reviews
    results_with_titles_merged = list(zip(sentiment_results, reviews_with_titles))

    # Create HTML content
    html_content = create_html_file(results_with_titles_merged)

    # Save HTML file
    save_html_file(html_content)

    # Display sentiment analysis results in the terminal window
    #display_results(sentiment_results,reviews)
    