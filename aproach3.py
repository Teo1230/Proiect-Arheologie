#the aproach with svc
import csv
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC  # Change to Support Vector Classifier
from sklearn.metrics import accuracy_score, classification_report
import spacy
import os
from spacy.tokens import Doc, Token
from sklearn.model_selection import GridSearchCV
import numpy as np

spacy.cli.download("en_core_web_sm")

nlp = spacy.load("en_core_web_sm")
Doc.set_extension("polarity", default=None)
Doc.set_extension("subjectivity", default=None)
Token.set_extension("is_negated", default=None)

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

def extract_structure(text):
    doc = nlp(text)
    structure_info = {
        "words": [token.text for token in doc],
        "lemmas": [token.lemma_ for token in doc],
        "pos_tags": [token.pos_ for token in doc],
        "pos_details": [(token.text, token.pos_, token.tag_) for token in doc],
        "dep_tags": [token.dep_ for token in doc],
        "named_entities": [(ent.text, ent.label_) for ent in doc.ents],
        "dependency_tree": [(token.text, token.dep_, token.head.text) for token in doc],
        "token_attributes": [(token.text, token.is_alpha, token.is_stop, token.is_punct) for token in doc],
        "sentiment": {"polarity": doc._.polarity, "subjectivity": doc._.subjectivity}  # Assuming sentiment analysis is available
    }
    return structure_info

def tune_svm(train_features, labels):
    param_grid = {'C': [1, 10, 100], 'gamma': [0.1, 0.01, 0.001], 'kernel': ['linear', 'rbf', 'poly']}

    model = SVC()

    grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')

    grid_search.fit(train_features, labels)

    print("Best Parameters (SVC):", grid_search.best_params_)

    return grid_search.best_estimator_

import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score, confusion_matrix

def evaluate(predictions, labels):
    print('******************************************')
    print('*********  EVALUATING PREDICTIONS ********')
    print('******************************************')
    print()

    accuracy = accuracy_score(labels, predictions)
    classification_rep = classification_report(labels, predictions)
    conf_mat = confusion_matrix(labels, predictions)

    print("Accuracy: {:.3f}".format(accuracy))
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", conf_mat)

    return accuracy, classification_rep, conf_mat


def main():
    data_mapping_train = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'crowd', 'train'))
    shared_task_posts_data = data_mapping_train['shared_task_posts']
    crowd_train_data = data_mapping_train['crowd_train']
    task_A_train_data = data_mapping_train['task_A_train']
    data_train = clustering_data(task_A_train_data, crowd_train_data, shared_task_posts_data)
    df_train = pd.DataFrame(data_train)

    data_mapping_test = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'crowd', 'test'), 'test')
    shared_test_posts_data = data_mapping_test['shared_task_posts']
    crowd_test_data = data_mapping_test['crowd_train']
    task_A_test_data = data_mapping_test['task_A_train']
    data_test = clustering_data(task_A_test_data, crowd_test_data, shared_test_posts_data)
    df_test = pd.DataFrame(data_test)

    df_train['structure_info'] = (df_train['post_title']+df_train['post_body']).apply(extract_structure)

    df_test['structure_info'] = (df_test['post_title']+df_test['post_body']).apply(extract_structure)

    tfidf_vectorizer = TfidfVectorizer(stop_words='english', max_features=5000)
    train_features = tfidf_vectorizer.fit_transform(df_train['post_title']+df_train['post_body'])
    test_features = tfidf_vectorizer.transform(df_test['post_title']+df_test['post_body'])

    tuned_svm_model = tune_svm(train_features, df_train['label'])

    tuned_svm_model.fit(train_features, df_train['label'])

    predictions_svm = tuned_svm_model.predict(test_features)
    accuracy, classification_rep, conf_matrix = evaluate(predictions_svm, df_test['label'])

if __name__ == "__main__":
    main()
