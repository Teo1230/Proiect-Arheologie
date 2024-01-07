import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier  # Added RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import spacy
import os
from spacy.tokens import Doc, Token
from sklearn.model_selection import GridSearchCV
import numpy as np



def load_csv(path_data):
    csv.field_size_limit(131072 * 10)

    expert_posts_columns = ['post_id', 'user_id', 'timestamp', 'subreddit', 'post_title', 'post_body']
    expert_posts_columns_file_path = os.path.join(path_data, "expert_posts.csv")


    expert_posts= []

    with open(expert_posts_columns_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=expert_posts_columns)
        next(reader, None)

        for row in reader:
            expert_posts.append(row)

    expert_columns = ['user_id', 'label']
    expert_path = os.path.join(path_data, "expert.csv")


    expert = []

    with open(expert_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=expert_columns)
        next(reader, None)

        for row in reader:
            expert.append(row)
    return {
        'posts': expert_posts,
        'users': expert,
    }

def path_find(file, path_data):
    path_data_set = os.getcwd()
    path_data_set = os.path.join(path_data_set, file)
    path_data_set = os.path.join(path_data_set, path_data)
    return path_data_set

def clustering_data(expert_posts, expert_users):
    data=[]
    for user in expert_users:
        user_id=user['user_id']
        for post in expert_posts:
            if post['user_id']==user_id:
                post['label']=user['label']
                data.append(post)
                break
    return data


def main():
    data_mapping_expert = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'expert'))
    expert_posts=data_mapping_expert['posts']
    expert_users=data_mapping_expert['users']
    expert_users=[user for user in expert_users if not user.get('label', '') == '']
    expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
    expert_posts=clustering_data(expert_posts,expert_users)
    data=pd.DataFrame(expert_posts)

    for index in range(len(data)):
        # print(data.user_id.iloc[index],data.post_body.iloc[index])
        print(index)
if __name__ == "__main__":
    main()