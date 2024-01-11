#varianta 1.1 folosesc yake sa extrag cuvinte specifice + pronume si adaug all_keywords ca param in tfidf_vectorizer 
#cea mai slaba varianta - nu parea ca face deloc bine

import shap
import yake
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
import numpy as np

from citire import data, df_test

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
