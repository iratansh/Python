from datetime import date, timedelta
import requests

def sentiment_analysis(stock):
    today = date.today()
    week_ago = today - timedelta(days=7)
    API = 'https://api.polygon.io/v2/aggs/ticker/'+ stock + '/range/1/day/' + str(week_ago) +'/' + str(today) + '?apiKey=kdqtByvNbe9w6AXHo3feX4F6K7eyQIJ1'
    news = requests.get(API)
    print(news.text)
    try:
        pass
    except Exception as e:
        print(e)
