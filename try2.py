#varianta 1.2 antrenez modelul pe shared_task apoi aplic pipeline ul direct pe cei 209 users din experts

import shap
import json
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np
from nltk import ngrams
from nltk.tokenize import word_tokenize
import re


from citire import data, df_test

mapare = {"a": 1, "b": -1, "c": -1, "d": -1}
df_test['label'] = df_test['label'].apply(lambda x: mapare[x])
data['label'] = data['label'].apply(lambda x: mapare[x])

tfidf_vectorizer = TfidfVectorizer(**{
    'min_df': 1,
    'max_features': None,
    'strip_accents': 'unicode',
    'analyzer': 'word',
    'token_pattern': r'\b[^\d\W]+\b',
    'ngram_range': (1, 3),
    'use_idf': True,
    'smooth_idf': True,
    'sublinear_tf': True,
    # 'vocabulary': all_keywords,
    # 'stop_words': stop_words#,
})

test_features = tfidf_vectorizer.fit_transform(df_test['post_title'] + df_test['post_body'])

model = LogisticRegression(class_weight='balanced')
model.fit(test_features, df_test['label'])

explainer = shap.LinearExplainer(model, test_features, feature_dependence="independent")
shap_values = explainer.shap_values(test_features)
X_test_array = test_features.toarray()

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())

from typing import List


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
        top_feature_indices = np.argsort(post_shap_values)[:7]
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


# output_data = {}

# for index, row in df_test.iterrows():
#     user_id = row['user_id']
#     post_id = row['post_id']

#     json_data = generate_json_output(user_id, post_id, shap_values, df_test, feature_names)

#     if user_id in output_data:
#         output_data[user_id]['posts'].append(json_data[user_id]['posts'][0])
#     else:
#         output_data[user_id] = json_data[user_id]


# output_file_path = 'outputSharedTask.json'
# with open(output_file_path, 'w') as json_file:
#     json.dump(output_data, json_file, indent=2)

# print(f"Output saved to {output_file_path}")

# output_file_path = 'outputSharedTask.json'

# with open(output_file_path, 'r') as json_file:
#     json_data = json.load(json_file)
# print(json.dumps(json_data, indent=2)[:10000])

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
