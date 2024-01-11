#varianta 1.2 antrenez modelul pe shared_task apoi aplic pipeline ul direct pe cei 209 users din experts
import csv
import pandas as pd
import shap
import json
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

tfidf_vectorizer = TfidfVectorizer(**{
    'min_df': 1,
    'max_features': None,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'\b[^\d\W]+\b',
    'ngram_range': (5, 10),
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': True,
    #'vocabulary': all_keywords,
    #'stop_words': stop_words#,
})

test_features = tfidf_vectorizer.fit_transform(df_test['post_title']+df_test['post_body'])

model = LogisticRegression(class_weight='balanced')
model.fit(test_features, df_test['label'])

explainer = shap.LinearExplainer(model, test_features, feature_dependence="independent") 
shap_values = explainer.shap_values(test_features)
X_test_array = test_features.toarray()

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

from typing import List
def verify_unique_strings(strings: List[str]):
    unique_strings = []

    for string in strings:
        i = 3
        substr = [string.split(" ")[i-3:i] for i in range(3,len(string))]
        substr = [i for i in substr if len(i)>=3]
        is_unique = all(all(" ".join(str1) not in other_string and other_string not in string for other_string in unique_strings) for str1 in substr)
        if is_unique:
            unique_strings.append(string)
    return unique_strings

def get_highlights(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]
        post_shap_values = shap_values[post_index]
        top_feature_indices = np.argsort(post_shap_values)[:10]
        top_features = ft_names[top_feature_indices]

        unique_top_features = verify_unique_strings(top_features)

        return unique_top_features
    else:
        return []
    
def generate_json_output(user_id, post_id, shap_values, dataset, ft_names):
    user_data = {
        "summarized_evidence": "Aggregating summary supporting assigned label",
        "posts": [
            {
                "post_id": post_id,
                "highlights": get_highlights(post_id, shap_values, dataset, ft_names)
            }
        ],
        "optional": [
            [
                "Optional field for reasoning chain data"
            ]
        ]
    }

    return {user_id: user_data}

output_data = {}

for index, row in df_test.iterrows():
    user_id = row['user_id']
    post_id = row['post_id']

    json_data = generate_json_output(user_id, post_id, shap_values, df_test, feature_names)
    
    if user_id in output_data:
        output_data[user_id]['posts'].append(json_data[user_id]['posts'][0])
    else:
        output_data[user_id] = json_data[user_id]


output_file_path = 'outputSharedTask.json'
with open(output_file_path, 'w') as json_file:
    json.dump(output_data, json_file, indent=2)

print(f"Output saved to {output_file_path}")

output_file_path = 'outputSharedTask.json'

with open(output_file_path, 'r') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:10000]) 

new_data = tfidf_vectorizer.transform(data['post_title'] + data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data, feature_dependence="independent")
shap_values_new = explainer_new.shap_values(new_data)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())

output_data_new = {}

for index, row in data.iterrows():
    user_id = row['user_id']
    post_id = row['post_id']

    json_data = generate_json_output(user_id, post_id, shap_values_new, data, feature_names_new)

    if user_id in output_data_new:
        output_data_new[user_id]['posts'].append(json_data[user_id]['posts'][0])
    else:
        output_data_new[user_id] = json_data[user_id]

output_file_path_new = 'outputExperts1.json'
with open(output_file_path_new, 'w') as json_file:
    json.dump(output_data_new, json_file, indent=2)

print(f"Output saved to {output_file_path_new}")

output_file_path_new = 'outputExperts1.json'

with open(output_file_path_new, 'r') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:10000])

