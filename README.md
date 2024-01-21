# The sentiment analysis of customer reviews

Are you interested to see what customers think about your comapny and your services but don't know how to do that? It's actually very simple and I can show you how to do that.

There are no special skills required as well. The Chat GPT and Github Copilot (optional) can help you to do the job!

1. First of all you need to find a website where cusotmers post their reviews. I found www.trustpilot.com one and it was good enough for my purpose. They had a company I was intersted in (Sun Life).

2. The next step was to collect these reviews. Since the number of reviews was not significant, I didn't use any database or other storage and simply loaded all of them into memory.

3. As per sentiment analysis, the [Hagging Face](https://huggingface.co/) was the simplest framework to use.

The overall architecture is below.

![Architecture diagram](/images/sentiment.png)

### Key findings

+ The browser Developer tools knowledge is a plus. It may bre required to identify labels used for HTML elements you need to scrape.

+ It is good to consider dynamic page detection if the program is supposed to run on a continious basis.

+ If customer's reviews are lengthy, the default model used by Hagging Face sentiment analysis pipeline can fail due to token size limitation. The one option is to select another model from [Huggig Face Hub](https://huggingface.co/docs/hub/models-the-hub) and another option is to cut off the reviews to fit the default model token size.

+ Don't trust Chat GPT 100%! Always double check by googling and apply critical thinking. E.g., Chat GPT suggested to use facebook/bart-large-cnn model from Hugging Face hub instead of default one to accomidate token size, but the model performed bad because it was not fine-tuned for sentiment analyis.

To review the results please go to the generated HTML file  [sentiment_analysis_results.html](./sentiment_analysis_results.html), download it, and open from the Downloads folder.

To download the file use the following button:

![Architecture diagram](/images/download.png)

The result was aligned with the overall rating on the website.
