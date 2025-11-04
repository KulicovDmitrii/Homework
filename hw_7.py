import requests
from bs4 import BeautifulSoup
import pandas as pd


def Analiz():
    all_data = []

    for page in range(1, 11):
        url = f"https://lifehacker.ru/topics/technology/?page={page}"
        print(f"Анализируем страницу {page}")

        response = requests.get(url)

        soup = BeautifulSoup(response.text, 'lxml')

        links = soup.find_all('a', class_='lh-small-article-card__link', href=True)

        article_urls = []
        for link in links:
                href = link['href']
                if href.startswith('/') and not href.startswith('/topics/technology/?page'):
                    full_url = 'https://lifehacker.ru' + href

                    article_urls.append(full_url)

        article_urls = list(set(article_urls))

        for article_url in article_urls:

                article_response = requests.get(article_url)

                article_soup = BeautifulSoup(article_response.text, 'lxml')

                title = article_soup.find('title')
                title_text = title.get_text() if title else "No title"

                paragraphs = article_soup.find_all('p')
                content = ' '.join([p.get_text().strip() for p in paragraphs])

                if content:
                    all_data.append({
                        'title': title_text,
                        'content': content,
                        'url': article_url
                    })
                print(f"Заголовок: {title_text}")
                print(f"Ссылка: {article_url}")


    return all_data

data = Analiz()

df = pd.DataFrame(data)
df.to_csv('lifehacker_simple.csv', index=False, encoding='utf-8')
print(f"Собрано {len(df)} статей")
