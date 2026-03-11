from io_utils import read_json, setup_logging, read_text

if __name__ == "__main__":
    setup_logging("pipeline.log")
    news_data = read_json("./data/raw/tmdb/news_articles_1.json")
    description_text = read_text("./data/raw/descriptions/description_1.txt")
    if news_data:
        print("News title:", news_data["title"])
    if description_text:
        print("First 20 characters of description:")
        print(description_text[:20])