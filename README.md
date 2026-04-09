# News_Media_Monitoring_Pipeline_UD

1. Describe your chosen project idea

The News Media Monitoring Pipeline collects news articles from multiple sources, processes the text (Im not sure if i should just process the text or also the images and videos) using natural language processing techniques, and analyzes trends to provide insights into current events, trends and media coverage.

2. Identify data sources (at least 2)

I am thinking about using NewsAPI(https://newsapi.org), NewsData.io(https://newsdata.io) and GDELT Project(https://www.gdeltproject.org)/GNews(https://gnews.io/) - i am not sure wether to use one or the other 

After some research i have found that NewsAPI, NewsData.io and GNews seem to be the most popular News APIs since they have a lot of data and offer good access to the data for developers, i have registered for the API keys and have since received them. 

GDelt is not typical since it doesnt have an API key i would have to use iut differently but it has a lot of global news data


3. List all data types you will handle

The pipeline will handle several types of unstructured data:

Text: news article titles, descriptions, and content retrieved from news APIs

Images: article images associated with news stories

Logs: pipeline execution logs and data collection records

The main data type analyzed in the project will be text, which will be processed using natural language processing techniques.

4. Expected challanges

Data Collection Reliability
News APIs and datasets may have rate limits or temporary downtime, which can interrupt the pipeline’s ability to consistently collect new articles.

Duplicate or Irrelevant Articles
Different news sources may publish the same story, which can result in duplicate data. Filtering and deduplicating articles may be necessary.

Text Processing Complexity
News articles often contain complex language, abbreviations, and named entities (people, locations, organizations), which can make natural language processing more challenging.

Large Data Volume
News sources generate large amounts of data daily, which may require efficient storage and processing methods.

Data Quality Issues
Some articles may have missing fields such as images, descriptions, or full text, which could affect analysis results.


6. Success criteria

The pipeline successfully collects news articles from at least two different sources.

The system can store and organize the collected data in a structured format (e.g., JSON or MongoDB).

Text processing techniques are applied to clean and analyze the article content.

The pipeline can identify basic trends or patterns in news coverage.

The results are visualized using charts or dashboards to provide insights about the collected data.