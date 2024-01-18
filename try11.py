#varianta 1.2 antrenez modelul pe shared_task apoi aplic pipeline ul direct pe cei 209 users din experts
# extrag caracteristicile si le pun in highlights si apoi propozitiile care contin acele expresii si le pun in summary
#extractive summary
# varianta in care folosesc range 2-4 in tfidf_vectorizer




import csv
import pandas as pd
import os
from nltk.tokenize import word_tokenize, sent_tokenize
from nltk.util import ngrams
import shap
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
import re

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
    'ngram_range': (2, 4),
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




def verify_unique_strings(strings):
    unique_strings = []

    for string in strings:
        string_copy = string.lower()
        tokens = word_tokenize(string_copy)
        substrings = [' '.join(gram) for gram in ngrams(tokens, 3) if all(len(word) >= 3 for word in gram)]

        is_unique = all(
            all(sub not in other_string and other_string not in sub for other_string in unique_strings) for sub in
            substrings)

        if is_unique:
            if any(contraction in string.lower() for contraction in [" ve ", " re ", " ll ", " m "]):
                string = re.sub(r'\b(ve|re|ll|m)\b', lambda match: "'" + match.group(1), string)
                string = string.replace(" '", "'")

            unique_strings.append(string)

    return unique_strings


def extract_sentence_with_feature(post_body, feature):
    post_body_copy = post_body.lower()
    post_body_copy = clean_sentence(post_body_copy)
    sentences = sent_tokenize(post_body_copy)
    matching_sentences = [sentence.strip() for sentence in sentences if feature.lower() in sentence.lower()]
    return matching_sentences


def clean_sentence(sentence):
    cleaned_sentence = re.sub(r'[^A-Za-z0-9\s.,\'"!?-]', '', sentence)
    return cleaned_sentence


def verify_sentences_existence(sentences, post_body):
    existing_sentences = [sentence for sentence in sentences if sentence.lower() in post_body.lower()]
    return existing_sentences


def get_summary(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]
        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:20]
            top_features = ft_names[top_feature_indices]
            unique_top_features = verify_unique_strings(top_features)

            extracted_sentences = []
            post_body = dataset.loc[post_index, 'post_body']

            for feature in unique_top_features:
                sentences = extract_sentence_with_feature(post_body, feature)
                extracted_sentences.extend(sentences)

            extracted_sentences = list(set(extracted_sentences))
            return extracted_sentences
    else:
        return []


def verify_top_features_existence(unique_top_features, post_body):
    verified_top_features = []

    for feature in unique_top_features:
        if feature.lower() in post_body.lower():
            verified_top_features.append(feature)

    return verified_top_features


def get_highlights(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]

        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:20]
            top_features = ft_names[top_feature_indices]

            unique_top_features = verify_unique_strings(top_features)
            verified_top_features = verify_top_features_existence(unique_top_features,
                                                                  dataset.loc[post_index, 'post_body'])

            return verified_top_features
        else:
            return []
    else:
        return []


def generate_json_output(user_id, post_id, shap_values, dataset, ft_names):
    extracted_sentences = get_summary(post_id, shap_values, dataset, ft_names)

    user_data = {
        "summarized_evidence": " ".join(extracted_sentences)[:300] if extracted_sentences else "",
        "posts": [
            {
                "post_id": post_id,
                "highlights": get_highlights(post_id, shap_values, dataset, ft_names)
            }
        ],
        "optional": [
            []
        ]
    }

    return {user_id: user_data}


filtered_data = data[data['user_id'].isin(all_user_ids_to_post_ids.keys())]
filtered_data = filtered_data[filtered_data['post_id'].isin(sum(all_user_ids_to_post_ids.values(), []))]

new_data_filtered = tfidf_vectorizer.transform(filtered_data['post_title'] + filtered_data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data_filtered, feature_dependence="independent")
shap_values_filtered = explainer_new.shap_values(new_data_filtered)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())
output_data_filtered = {}

for user_id, user_posts in filtered_data.groupby('user_id'):
    user_data = {
        "summarized_evidence": "",
        "posts": [],
        "optional": [[]]
    }

    for index, row in user_posts.iterrows():
        post_id = row['post_id']

        extracted_sentences = get_summary(post_id, shap_values_filtered, filtered_data, feature_names_new)

        user_data["summarized_evidence"] += " ".join(extracted_sentences)[:300] + " " if extracted_sentences else ""

        user_data["posts"].append({
            "post_id": post_id,
            "highlights": get_highlights(post_id, shap_values_filtered, filtered_data, feature_names_new)
        })

    output_data_filtered[user_id] = user_data

output_file_path_filtered = 'outputExpertEextractiveSummary_filtered.json'
with open(output_file_path_filtered, 'w') as json_file:
    json.dump(output_data_filtered, json_file, indent=2)

with open(output_file_path_filtered, 'r') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:10000])