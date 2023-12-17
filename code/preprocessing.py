import requests
import json
import csv
import time
import pandas as pd
import numpy as np
import os
import nltk
from nltk.corpus import stopwords


def split_dataset(panel_data):
    # Load the panel dataset

    # Convert the year column to a datetime object
    panel_data['pub_date'] = pd.to_datetime(panel_data['pub_date'], format='%Y-%m-%d')

    # Subset the data frame based on the desired time periods
    obama_2013_2017 = panel_data[(panel_data['pub_date'] >= '2013-01-20') & (panel_data['pub_date'] <= '2017-01-20')]
    trump_2017_2021 = panel_data[(panel_data['pub_date'] >= '2017-01-21') & (panel_data['pub_date'] <= '2021-01-20')]

    obama_headlines = obama_2013_2017.loc[obama_2013_2017['headline'].str.contains('Obama').fillna(False)]
    trump_headlines = trump_2017_2021.loc[trump_2017_2021['headline'].str.contains('Trump').fillna(False)]



    # Save the data frames as separate CSV files
    obama_headlines.to_csv('obama.csv', index=False)

    trump_headlines.to_csv('trump.csv', index=False)



def create_csv():
    # set the path to the folder containing the CSV files
    path_to_folder = '/Users/oliverkanders/Desktop/mlEcon/project-2/nyt'

    # create an empty list to store the data frames
    cleaned_data = pd.DataFrame()

    # loop through the files in the folder
    for filename in os.listdir(path_to_folder):
        if filename.endswith('.csv'):
            # read in the CSV file
            file_path = os.path.join(path_to_folder, filename)
            df = pd.read_csv(file_path)


            # identify 6 features to keep
            healdine = ['pub_date','headline']

            #select only headline to keep, drop the rest
            df = df.loc[:, healdine]     
    

            # append the data frame to the list
            cleaned_data = cleaned_data.append(df)

    # concatenate the data frames in the list into a single data frame
    #combined_df = pd.concat(df_list, axis=0)

    # print the combined data frame
    cleaned_data.sort_values('pub_date', inplace=True)

    print(cleaned_data)

    # specify the file name for the CSV file
    file_name = 'nyt.csv'

    # save the data frame as a CSV file in the same folder as the script
    cleaned_data.to_csv(file_name, index=False)

    print(f"The data frame has been saved as {file_name} in the same folder.")
    return cleaned_data




def fetch_articles(api_key, politics, begin_date, end_date, num_articles):
    base_url = 'https://api.nytimes.com/svc/search/v2/articlesearch.json'
    articles = []
    #this is for 10 article pages
    for page in range((num_articles // 10) + 1):
        params = {
            'api-key': api_key,
            'begin_date': begin_date,
            'end_date': end_date,
            'page': page,
            'q': politics,
            'sort': 'newest'
        }
        response = requests.get(base_url, params=params)
        if response.status_code == 200:
            articles += response.json()['response']['docs']
            time.sleep(5)  # wait one second before the next request
        else:
            print(f'Error fetching articles: {response.status_code}')
            return None
    return {'response': {'docs': articles[:num_articles]}}


def extract_text(article_data):
    articles = []
    for article in article_data['response']['docs']:
        headline = article['headline']['main']
        snippet = article['snippet']
        text = headline + ' ' + snippet
        articles.append(text)
    return articles

def write_to_csv(articles_text, filename):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(['Text'])
        for text in articles_text:
            writer.writerow([text])

def main():

    df = create_csv()
    split_dataset(df)
    #api_key = 'Ygj0TqQZf4VtGm25D1o7GDPUJ6MTcWFb'

    # Fetch pre-Trump articles
    #pre_trump_begin_date = '20090120'  # Obama's inauguration date
    #pre_trump_end_date = '20170119'  # The day before Trump's inauguration
    #politics = 'Barack Obama OR President (scandal OR controversy OR investigation)'
    #pre_trump_articles = fetch_articles(api_key, politics, pre_trump_begin_date, pre_trump_end_date, 100)

    # Fetch post-Trump articles
    #post_trump_begin_date = '20170120'  # Trump's inauguration date
    #post_trump_end_date = '20210416'  # You can set this to today's date or any date during/after Trump's presidency            
    #query_trump = 'Donald Trump OR President (scandal OR controversy OR impeachment OR investigation)'

    #post_trump_articles = fetch_articles(api_key, query_trump, post_trump_begin_date, post_trump_end_date, 100)

    # Print the fetched articles
    #print('Pre-Trump articles:')
    #print(json.dumps(pre_trump_articles, indent=2))

    #print('Post-Trump articles:')
    #print(json.dumps(post_trump_articles, indent=2))

    #pre_trump_articles_text = extract_text(pre_trump_articles)
    #post_trump_articles_text = extract_text(post_trump_articles) 

    #write_to_csv(pre_trump_articles_text, 'pre_trump_articles.csv')
    #write_to_csv(post_trump_articles_text, 'post_trump_articles.csv')



if __name__ == '__main__':
    main()
