#varianta 1.1 folosesc yake sa extrag cuvinte specifice + pronume si adaug all_keywords ca param in tfidf_vectorizer 
#cea mai slaba varianta - nu parea ca face deloc bine

import csv
import pandas as pd
import shap
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import os
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

data_mapping_expert = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data(expert_posts,expert_users)
data=pd.DataFrame(expert_posts)                     #cei 209 de experti

def load_csv(path_data, test=None):
    csv.field_size_limit(131072 * 10)

    shared_task_columns = ['post_id', 'user_id', 'timestamp', 'subreddit', 'post_title', 'post_body']
    if test is None:
        shared_task_file_path = os.path.join(path_data, "shared_task_posts.csv")
    else:
        shared_task_file_path = os.path.join(path_data, "shared_task_posts_test.csv")

    shared_task_posts = []

    with open(shared_task_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=shared_task_columns)
        next(reader, None)

        for row in reader:
            shared_task_posts.append(row)

    crowd_columns = ['user_id', 'label']
    if test is None:
        crowd_file_path = os.path.join(path_data, "crowd_train.csv")
    else:
        crowd_file_path = os.path.join(path_data, "crowd_test.csv")

    crowd_train = []

    with open(crowd_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=crowd_columns)
        next(reader, None)

        for row in reader:
            crowd_train.append(row)

    task_A_columns = ['post_id', 'user_id', 'subreddit']
    if test is None:
        task_A_file_path = os.path.join(path_data, "task_A_train.posts.csv")
    else:
        task_A_file_path = os.path.join(path_data, "task_A_test.posts.csv")

    task_A_train = []

    with open(task_A_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=task_A_columns)
        next(reader, None)

        for row in reader:
            task_A_train.append(row)

    return {
        'shared_task_posts': shared_task_posts,
        'crowd_train': crowd_train,
        'task_A_train': task_A_train
    }

def clustering_data(task_A_data, crowd_data, shared_task_posts_data):
    for task in task_A_data:
        for shared_task in shared_task_posts_data:
            if task['post_id'] == shared_task['post_id']:
                task['timestamp'] = shared_task['timestamp']
                task['subreddit'] = shared_task['subreddit']
                task['post_title'] = shared_task['post_title']
                task['post_body'] = shared_task['post_body']
                break
        for crowd in crowd_data:
            if task['user_id'] == crowd['user_id']:
                task['label'] = crowd['label']
                break
    return task_A_data

def path_find(file, path_data, path_train):
    path_data_set = os.getcwd()
    path_data_set = os.path.join(path_data_set, file)
    path_data_set = os.path.join(path_data_set, path_data)
    path_data_set = os.path.join(path_data_set, path_train)
    return path_data_set

data_mapping_test = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'crowd', 'test'), 'test')
shared_test_posts_data = data_mapping_test['shared_task_posts']
crowd_test_data = data_mapping_test['crowd_train']
task_A_test_data = data_mapping_test['task_A_train']
data_test = clustering_data(task_A_test_data, crowd_test_data, shared_test_posts_data)
df_test = pd.DataFrame(data_test)

mapare = {"a":1, "b":-1, "c": -1, "d": -1}
df_test['label'] = df_test['label'].apply(lambda x: mapare[x])
data['label'] = data['label'].apply(lambda x: mapare[x])

def extract_keywords(text):
    kw_extractor = yake.KeywordExtractor()
    keywords = kw_extractor.extract_keywords(text)
    return [keyword[0] for keyword in keywords]
    
df_test['keywords'] = df_test['post_title'].apply(extract_keywords) + df_test['post_body'].apply(extract_keywords)
all_keywords = set(keyword.lower() for keywords_list in df_test['keywords'] for keyword in keywords_list)
pronouns = specified_keywords = {"i", "you", "he", "she", "it", "we", "they", "me", "him", "her", "us", "them", "my", "your", "his", "its", "our", "their", "mine", "yours", "hers", "ours", "theirs"}
all_keywords = all_keywords.union(pronouns)


tfidf_vectorizer = TfidfVectorizer(**{
    'min_df': 1,
    'max_features': None,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'\b[^\d\W]+\b',
    'ngram_range': (1,5),
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': True,
    'vocabulary': all_keywords,
    #'stop_words': stop_words#,
})

test_features = tfidf_vectorizer.fit_transform(df_test['post_title']+df_test['post_body'])

model = LogisticRegression(class_weight='balanced')
model.fit(test_features, df_test['label'])

new_data = tfidf_vectorizer.transform(data['post_title'] + data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data, feature_dependence="independent")
shap_values_new = explainer_new.shap_values(new_data)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())
expressions = set(feature_names_new[i] for i in range(len(feature_names_new)) if shap_values_new[1][i] < 0)

print(expressions)
