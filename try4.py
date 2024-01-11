#varianta 1.4 - extrag lista de expresii cu impact mare negativ din shared_task si le caut in experts

import shap
from sklearn.feature_extraction.text import TfidfVectorizer
import json
from sklearn.linear_model import LogisticRegression
import numpy as np

from citire import data, df_test

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

feature_names = np.array(tfidf_vectorizer.get_feature_names_out())
unique_expressions = set(feature_names[i] for i in np.where(shap_values[1] < 0)[0])
#print(unique_expressions)

def get_highligh(post_id, dataset, unique_expressions):
    matching_rows = dataset[dataset['post_id'] == post_id]
    if not matching_rows.empty:
        post_index = matching_rows.index[0]
        post_text = matching_rows['post_body'].values[0]

        highlights = [expression for expression in unique_expressions if expression in post_text]

        return highlights
    else:
        return []
    

def generate_json_outp(user_id, post_id, dataset, unique_expressions):
    user_data = {
        "summarized_evidence": "Aggregating summary supporting assigned label",
        "posts": [
            {
                "post_id": post_id,
                "highlights": get_highligh(post_id, dataset, unique_expressions)
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
for post in expert_posts:
    user_id = post['user_id']
    post_id = post['post_id']

    json_data = generate_json_outp(user_id, post_id, data, unique_expressions)
    
    if user_id in output_data:
        output_data[user_id]['posts'].append(json_data[user_id]['posts'][0])
    else:
        output_data[user_id] = json_data[user_id]

output_file_path = 'output_strings.json'
with open(output_file_path, 'w') as json_file:
    json.dump(output_data, json_file, indent=2)

print(f"Output saved to {output_file_path}")

output_file_path = 'output_strings.json'

with open(output_file_path, 'r') as json_file:
    json_data = json.load(json_file)

print(json.dumps(json_data, indent=2)[:10000]) 