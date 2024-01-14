#varianta 1.3 antrenez direct pe experts si apoi folosesc shap ca sa extrag expresiile importante

import csv
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from sklearn.linear_model import LogisticRegression
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
            if post['user_id']==user_id and post['subreddit']=='SuicideWatch':
                post['label']=user['label']
                data.append(post)
                break
    print(len(data))
    return data

data_mapping_expert = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
#expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data(expert_posts,expert_users)
data=pd.DataFrame(expert_posts)  

mapare = {"a":1, "b":-1, "c": -1, "d": -1}
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
})

test_features = tfidf_vectorizer.fit_transform(data['post_title']+data['post_body'])

model = LogisticRegression(class_weight='balanced')
model.fit(test_features, data['label'])

explainer = shap.LinearExplainer(model, test_features, feature_dependence="independent") 
shap_values = explainer.shap_values(test_features)
X_test_array = test_features.toarray()

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
# unique_expressions = set(feature_names[i] for i in np.where(shap_values[1] < 0)[0])
# print(unique_expressions)

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

for index, row in data.iterrows():
    user_id = row['user_id']
    post_id = row['post_id']

    json_data = generate_json_output(user_id, post_id, shap_values, data, feature_names)
    
    if user_id in output_data:
        output_data[user_id]['posts'].append(json_data[user_id]['posts'][0])
    else:
        output_data[user_id] = json_data[user_id]

output_file_path = 'outputExperts2.json'
with open(output_file_path, 'w') as json_file:
    json.dump(output_data, json_file, indent=2)

print(f"Output saved to {output_file_path}")

output_file_path = 'outputExperts2.json'

with open(output_file_path, 'r') as json_file:
    json_data = json.load(json_file)

print(json.dumps(json_data, indent=2)[:10000]) 