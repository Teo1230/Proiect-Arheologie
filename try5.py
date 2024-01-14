#extrag caracteristicile si apoi propozitiile care contin acele expresii
#varianta in care folosesc range 5-10 in tfidf_vectorizer

import csv
import pandas as pd
import shap
from sklearn.feature_extraction.text import TfidfVectorizer
import os
import json
from sklearn.linear_model import LogisticRegression
import numpy as np
import re
import spacy
from nltk import ngrams
import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')

nlp = spacy.load("en_core_web_sm")



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
    return data

data_mapping_expert = load_csv(path_find('../umd_reddit_suicidewatch_dataset_v2', 'expert'))
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
unique_expressions = set(feature_names[i] for i in np.where(shap_values[1] < 0)[0])
#print(unique_expressions)


def verify_unique_strings(strings):
    unique_strings = []

    for string in strings:
        string_copy = string.lower()  # Create a copy to avoid altering the original string
        tokens = word_tokenize(string_copy)
        substrings = [' '.join(gram) for gram in ngrams(tokens, 3) if all(len(word) >= 3 for word in gram)]

        is_unique = all(
            all(sub not in other_string and other_string not in sub for other_string in unique_strings) for sub in
            substrings)

        if is_unique:
            unique_strings.append(string)

    return unique_strings


def extract_sentence_with_feature(post_body, feature):
    post_body_copy = post_body.lower()
    post_body_copy = clean_sentence(post_body_copy)
    tokens = word_tokenize(post_body_copy)
    sentences = [sentence.strip() for sentence in re.split(r'[.!?]', post_body_copy) if
                 feature.lower() in sentence.lower()]
    return sentences


def clean_sentence(sentence):
    cleaned_sentence = re.sub(r'[^A-Za-z0-9\s.,\'"!?]', '', sentence)
    return cleaned_sentence


def get_highlights(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]
        post_shap_values = shap_values[post_index]
        top_feature_indices = np.argsort(post_shap_values)[:15]
        top_features = ft_names[top_feature_indices]
        unique_top_features = verify_unique_strings(top_features)

        highlights = []
        post_body = dataset.loc[post_index, 'post_body']

        for feature in unique_top_features:
            sentences = extract_sentence_with_feature(post_body, feature)
            highlights.extend(sentences)

        highlights = list(set(highlights))

        return highlights
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

    if not row['post_body'].strip():
        print("Post body is empty for this row.")

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

print(json.dumps(json_data, indent=2)[:20000])