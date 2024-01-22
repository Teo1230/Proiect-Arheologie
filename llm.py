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
    data={}
    all_user_ids_to_post_ids = {
    "19041": ["1ftvgt", "2j2t7i"],
    "19239": ["1wj80m"],
    "19788": ["36eh73", "3am642"],
    "48665": ["ev6tp"],
    "1523": ["3ejsiv"],
    "6939": ["17r00i", "3fuxue"],
    "6998": ["3c83gc"],
    "7302": ["3gcaik"],
    "10090": ["22l8w4", "2mb3ku"],
    "11388": ["2m9gm4"],
    "12559": ["2sq2me"],
    "14958": ["1shwnv"],
    "15807": ["l5jxp"],
    "18608": ["2v0w14"],
    "18639": ["2f7wj4"],
    "20277": ["3hy1hk", "3i9p8b"],
    "21653": ["16xkll"],
    "23080": ["1gceob", "1t7se3"],
    "23341": ["3dthkh"],
    "23400": ["1y298o"],
    "24003": ["2jo2q0"],
    "24331": ["2zienb"],
    "27262": ["2u5z30"],
    "29356": ["26nqsc"],
    "30179": ["2evvoo"],
    "34032": ["1otrx6"],
    "34464": ["35lq71"],
    "34502": ["12umtk"],
    "37770": ["3f9nd4", "3gk55c"],
    "38531": ["2qlffz"],
    "39656": ["30reha"],
    "39832": ["1xpagm", "2017wz"],
    "40671": ["357o7f"],
    "41651": ["2scmvs", "2u91iq"],
    "43030": ["2fwbqi"],
    "43328": ["2x08xn"],
    "44826": ["1k3agb"],
    "46267": ["2nmehc"],
    "49038": ["2gt6wk", "2hnma9"],
    "11360": ["2xrnry"],
    "13321": ["3ic4t8"],
    "14075": ["32b7zk", "33kh5m", "35ylua"],
    "16852": ["1631gw"],
    "16867": ["30xti7", "32jo2k"],
    "18321": ["21mm33", "241gpm"],
    "19250": ["3gizbj"],
    "21926": ["1ph5yt"],
    "28817": ["36xc3s", "383n1f", "3cq8kt"],
    "29142": ["38bc2o"],
    "31972": ["2iz5y7"],
    "33368": ["2f27hy"],
    "33824": ["33r8b6"],
    "34268": ["2z2q1f"],
    "37759": ["3152g6"],
    "46616": ["10llu1"],
    "46771": ["3896jf"],
    "24231": ["23tvqs"],
    "35822": ["24u2gv"],
    "44554": ["2jfo9n", "2onplv"],
    "47912": ["thn6n"],
    "3434": ["3h0z05"],
    "5916": ["280apu"],
    "8368": ["1azxoz"],
    "8547": ["39akaq"],
    "10940": ["2h4b4q"],
    "13753": ["2vcol2"],
    "14272": ["yaqxi"],
    "14302": ["2b3y50", "2booi6"],
    "14680": ["16d5de"],
    "15722": ["31dz04"],
    "16366": ["3dv1gk"],
    "16596": ["2rp5bb", "319ioh"],
    "16673": ["2uquru"],
    "17606": ["1tplbc"],
    "18616": ["xfyos"],
    "19510": ["3gx2sj"],
    "25320": ["2aqb2a", "2kvmof"],
    "27214": ["mxci8", "13c9hl", "1455fd"],
    "31509": ["2c7y41"],
    "31683": ["3gjkm6"],
    "32490": ["2h1vke"],
    "32714": ["35miyb"],
    "34078": ["2802g5"],
    "36117": ["1mjszi", "2d3de4"],
    "43263": ["25rnwu"],
    "43701": ["3hxe6j"],
    "44656": ["278mzu"],
    "44846": ["2yek5b"],
    "45909": ["2ng6fr"],
    "46258": ["1wb4qr", "2wsqbn"],
    "46530": ["20d48v"],
    "47012": ["37jcax"],
    "47614": ["pnkh4", "15j1ft"],
    "47899": ["19jxq1"],
    "50326": ["3akx6o"],
    "3058": ["3iagg6"],
    "3928": ["32i9nz"],
    "5224": ["2a7wms"],
    "14737": ["3fvr4l"],
    "17820": ["k76ov"],
    "19058": ["2zqq2c", "32o53l"],
    "19916": ["2e5zho"],
    "20007": ["1w6g6u"],
    "29501": ["1getay", "1gwlbb"],
    "30593": ["2uafbk"],
    "32858": ["2y7a6g", "3ci2ni"],
    "38515": ["1pl51l"],
    "42410": ["1zhg6p", "224i1n", "227n97"],
    "47251": ["2dq4gx"],
    "47572": ["2tfdb4", "3ff64x"],
    "50337": ["3f7iqw"],
    "51395": ["189w0w"],
    "123": ["2v6c9q", "3e8si7"],
    "3697": ["2ff664"],
    "5616": ["g7f7g"],
    "6303": ["20kbpj"],
    "6572": ["2xa7jl"],
    "243": ["2j44ow"],
    "777": ["1kpxiu"],
    "836": ["1d7nab", "1unus3"],
    "1270": ["2dv5mw"],
    "373": ["2xe7hp", "3bhds2", "3d2ttx"],
    "899": ["3h9o7o", "3hh29z"],
    "1264": ["vkxfb", "2nledl"],
    "2421": ["2il6xf"],
}

    for user in expert_users:
        user_id=user['user_id']
        for post in expert_posts:
            if user_id in all_user_ids_to_post_ids.keys() and post['user_id']==user_id and post['subreddit']=='SuicideWatch' :
                post['label']=user['label']
                if user_id in data:
                    data[user_id].append(post)
                else:
                    data[user_id]=[]
                    data[user_id].append(post)

    return data

data_mapping_expert = load_csv(path_find('umd_reddit_suicidewatch_dataset_v2', 'expert'))
expert_posts=data_mapping_expert['posts']
expert_users=data_mapping_expert['users']
expert_users=[user for user in expert_users if not user.get('label', '') == '']
expert_users=[user for user in expert_users if not user.get('label', '') == 'a']
expert_posts=clustering_data(expert_posts,expert_users)



template = """Question: {question}

Answer: Let's work this out in a step by step way to be sure we have the right answer."""

prompt = PromptTemplate(template=template, input_variables=["question"])
callback_manager = CallbackManager([StreamingStdOutCallbackHandler()])
llm = LlamaCpp(
    model_path="openhermes-2.5-mistral-7b.Q4_K_M.gguf",
    temperature=0.75,
    max_tokens=2000,
    top_p=1,
    n_ctx=32000,
    callback_manager=callback_manager,
    verbose=True,  # Verbose is required to pass to the callback manager
)
llm_chain = LLMChain(prompt=prompt, llm=llm)
def extract_optional_content(highlights):
    try:
        extracted_content = re.findall(r'"([^"]*)"', highlights)
        extracted_content = [i.rstrip('.,') for i in extracted_content]

        return extracted_content
    except KeyError:
        return None  # Fix the return statement

def remove_duplicate_highlights(highlights):
    sorted_highlights = sorted(highlights, key=len, reverse=False)
    for i in range(len(sorted_highlights)-1):
        for j in range(i + 1, len(sorted_highlights)):
            if sorted_highlights[i] in sorted_highlights[j]:
                sorted_highlights[i] = ""
                break
    return [highlight for highlight in sorted_highlights if highlight]


for user_id in expert_posts.keys():
    for post in expert_posts[user_id]:
        post_body = post['post_body']
        if post_body == '':
            post_body = post['post_title']
        for _ in range(2):
            question = f"Provide sequences of text that indicate that this person is suicidal?\n\nPost Body: {post_body}"
            result = llm_chain.run(question)
            if result=="":
                result = llm_chain.run(question)
            limited_result = ' '.join(result.split()[:300])
            if 'highlights' not in post:
                post['highlights']=""
            post['highlights'] = post['highlights']+limited_result
        post['highlights']= remove_duplicate_highlights(extract_optional_content(post['highlights']))
        ls = []
        original_content = (post['post_body'] + post['post_title']).lower()
        for sentece in  post['highlights']:
            sentece = sentece.lower()
            if sentece != "it" and sentece != " and " and sentece != ", and " and sentece != '' and sentece != ", " and sentece != " " and sentece in original_content and sentece not in ls:
                ls.append(sentece)
        post['highlights']=ls
    # break

for user_id in expert_posts.keys():
    content_body=""
    user_id='10090'
    for post in expert_posts[user_id]:
        content_body=content_body+post['post_body']
    question = f"As a psychologist and expert therapist, summarize the content by identifying any indications of suicidal thoughts. Provide evidence from the text to support your analysis.\n\nPost Body: {content_body} Analyze"
    result = llm(question)
    result = " ".join(result.split()[:300])
    find =result.find("\n\n")
    if find != -1:
        result = result[find:]
        result = result.replace('\n\n', '')

    if result=="":
        result = llm(question)
        find = result.find("\n\n")
        if find != -1:
            result = result[find:]  # Corrected slicing
            result = result.replace('\n\n', '')

    result = " ".join(result.split()[:300]).lower()
    post['summarized_evidence']=result
    for post in expert_posts[user_id]:
        extract = extract_optional_content(result)
        ls = post['highlights']
        original_content = (post['post_body'] + post['post_title']).lower()
        for sentece in extract:
            sentece = sentece.lower()
            if sentece != "it" and sentece != " and " and sentece != ", and " and sentece != '' and sentece != ", " and sentece != " " and sentece in original_content and sentece not in ls:
                ls.append(sentece)

        post['highlights'] = remove_duplicate_highlights(ls)
    # break

with open("summary_combined_body_post.json", "w") as file:
    json.dump(expert_posts, file,indent=4)
