import csv
import json
import re

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
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.streaming_stdout import StreamingStdOutCallbackHandler
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import LlamaCpp


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
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data(expert_posts,expert_users)
data=pd.DataFrame(expert_posts)

template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])

# Callbacks support token-wise streaming
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])

# Make sure the model path is correct for your system!
llm = LlamaCpp(
    model_path="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_ctx=32000,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
# there might be an issue with the chain
# https://github.com/abetlen/llama-cpp-python/issues/944
llm_chain = LLMChain(prompt=prompt, llm=llm)

highlights=[]
for post in expert_posts:
    user_id= post['user_id']
    post_body = post['post_body']
    if post_body=='':
        post_body = post['post_title']

    question = f"Provide sequences of text that indicate that this person is suicidal?\n\nPost Body: {post_body}"
    result = llm_chain.run(question)
    data={"user_id":user_id, "highlights":result}
    highlights.append(data)
    # break

summary = []
for post in expert_posts:
    data = {}
    user_id = post['user_id']
    post_body = post['post_body']

    if post_body == '':
        post_body = post['post_title']

    question = f"As a psychologist and expert therapist, summarize the content by identifying any indications of suicidal thoughts. Provide evidence from the text to support your analysis.\n\nPost Body: {post_body}"
    result = llm(' '.join(question.split(' ')[:312]))

    p = {}
    p['post_id'] = post['post_id']
    p['highlights'] = ''

    data[user_id] = {
        'summarized_evidence': result,
        'posts': [p],
        'optional': [[]]
    }
    summary.append(data)
    # break

def extract_optional_content(data):
    try:
        user_id = data["user_id"]
        optional_content = data["highlights"]

        # Extract content within double quotes using regular expression
        extracted_content = re.findall(r'"([^"]*)"', optional_content)

        return user_id, extracted_content
    except KeyError:
        return None, None
print(highlights)
for data in highlights:
        user_id, extracted_content = extract_optional_content(data)
    # Print the extracted content
        if user_id is not None and extracted_content is not None:
            # print(f"User ID: {user_id}")
            # print(f"Extracted Content within Double Quotes:\n{extracted_content}")
            data['highlights']=extracted_content
        else:
            print("Error extracting data from JSON.")
print(highlights)

print(summary)
for i in summary:
    for i3 in i.keys():
        user_id=i3

    for j in highlights:
        if j['user_id'] == user_id:
            i[user_id]['posts'][0]['highlights']=j['highlights']
            break
print(summary)

for data in summary:
    for key, value in data.items():
        find = value['summarized_evidence'].find("\n\n")
        print(find)
        if find != -1:
            content = value['summarized_evidence']
            data[key]['summarized_evidence'] = content[find:]

with open("filter_submission.json", "w") as file:
     json.dump(summary, file,indent=4)