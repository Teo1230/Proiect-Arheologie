def verify_unique_strings(strings):
    unique_strings = []

    for string in strings:
        string_copy = string
        tokens = word_tokenize(string_copy)
        substrings = [' '.join(gram) for gram in ngrams(tokens, 3) if all(len(word) >= 3 for word in gram)]

        is_unique = all(
            all(sub not in other_string and other_string not in sub for other_string in unique_strings) for sub in
            substrings)

        if is_unique:
            unique_strings.append(string)

    return unique_strings


def extract_sentence_with_feature(post_body, feature):
    post_body_copy = post_body
    post_body_copy = clean_sentence(post_body_copy)
    tokens = word_tokenize(post_body_copy)
    sentences = [sentence.strip() for sentence in re.split(r'[.!?]', post_body_copy) if
                 feature in sentence]
    return sentences


def clean_sentence(sentence):
    cleaned_sentence = re.sub(r'[^A-Za-z0-9\s.,\'"!?-_[]()]', '', sentence)
    return cleaned_sentence


def get_summarized_evidence(post_id, shap_values, dataset, ft_names):
    matching_rows = dataset[dataset['post_id'] == post_id]

    if not matching_rows.empty:
        post_index = matching_rows.index[0]

        if post_index < len(shap_values):
            post_shap_values = shap_values[post_index]
            top_feature_indices = np.argsort(post_shap_values)[:50]
            top_features = ft_names[top_feature_indices]
            unique_top_features = verify_unique_strings(top_features)

            summarized_evidence = []
            post_body = dataset.loc[post_index, 'post_body']

            for feature in unique_top_features:
                sentences = extract_sentence_with_feature(post_body, feature)
                summarized_evidence.extend(sentences)

            summarized_evidence = list(set(summarized_evidence))

            return summarized_evidence
    else:
        return []


def limit_words(text, word_limit):
    words = text.split()
    if len(words) <= word_limit:
        return text
    else:
        return ' '.join(words[:word_limit])


def verify_unique_strings_highlights(strings):
    unique_strings = set()

    for string in strings:
        tokens = word_tokenize(string.lower())

        valid_tokens = [word for word in tokens if len(word) >= 3]

        reconstructed_string = ' '.join(valid_tokens)
        if reconstructed_string not in unique_strings:
            if any(contraction in string.lower() for contraction in [" ve ", " re ", " ll ", " m ", " t ", " s "]):
                reconstructed_string = re.sub(r'\b(ve|re|ll|m)\b', lambda match: "'" + match.group(1),
                                              reconstructed_string)
                reconstructed_string = reconstructed_string.replace(" '", "'")

            unique_strings.add(reconstructed_string)

    return list(unique_strings)


def verify_top_features_existence(unique_top_features, post_body):
    verified_top_features = []

    for feature in unique_top_features:
        if feature.lower() in post_body.lower():
            verified_top_features.append(feature)

    return verified_top_features


# def get_highlights(post_id, shap_values, dataset, ft_names, context_words=2):
#     matching_rows = dataset[dataset['post_id'] == post_id]

#     if not matching_rows.empty:
#         post_index = matching_rows.index[0]

#         if post_index < len(shap_values):
#             post_shap_values = shap_values[post_index]
#             top_feature_indices = np.argsort(post_shap_values)[:20]
#             top_features = ft_names[top_feature_indices]

#             unique_top_features = verify_unique_strings_highlights(top_features)
#             verified_top_features = verify_top_features_existence(unique_top_features, dataset.loc[post_index, 'post_body'])

#             alternative = verified_top_features
#             highlights = []

#             for feature in verified_top_features:
#                 feature_lower = feature.lower()
#                 feature_index = dataset.loc[post_index, 'post_body'].lower().find(feature_lower)

#                 if feature_index != -1:
#                     start_index = max(0, feature_index - context_words * 2)
#                     end_index = min(len(dataset.loc[post_index, 'post_body']), feature_index + len(feature_lower) + context_words * 2)

#                     context_text = dataset.loc[post_index, 'post_body'][start_index:end_index].strip()
#                     highlights.append(context_text)

#             return highlights
#         else:
#             return alternative
#     else:
#         return []

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


def generate_json_output(user_id, post_id, shap_values, dataset, ft_names, word_limit=300):
    extracted_sentences = get_summarized_evidence(post_id, shap_values, dataset, ft_names)

    user_data = {
        "summarized_evidence": "",
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

    for sentence in extracted_sentences:
        user_data["summarized_evidence"] += sentence + " "

    user_data["summarized_evidence"] = limit_words(user_data["summarized_evidence"], word_limit)

    return {user_id: user_data}


new_data = tfidf_vectorizer.transform(data['post_title'] + data['post_body']).toarray()
explainer_new = shap.LinearExplainer(model, new_data, feature_dependence="independent")
shap_values_new = explainer_new.shap_values(new_data)

feature_names_new = np.array(tfidf_vectorizer.get_feature_names_out())

output_data_new = {}

for user_id, user_posts in filtered_data.groupby('user_id'):
    user_data = {
        "summarized_evidence": "",
        "posts": [],
        "optional": [[]]
    }

    for index, row in user_posts.iterrows():
        post_id = row['post_id']

        extracted_sentences = get_summarized_evidence(post_id, shap_values_new, data, feature_names_new)

        user_data["summarized_evidence"] += ". ".join(extracted_sentences) + " " if extracted_sentences else ""

        user_data["posts"].append({
            "post_id": post_id,
            "highlights": get_highlights(post_id, shap_values_new, data, feature_names_new)
        })

    output_data_new[user_id] = user_data

users_to_keep = set(all_user_ids_to_post_ids.keys())

users_to_delete = [user_id for user_id in output_data_new.keys() if user_id not in users_to_keep]

for user_id in users_to_delete:
    del output_data_new[user_id]

output_file_path_new = 'submission1.json'
with open(output_file_path_new, 'w', encoding='utf-8') as json_file:
    json.dump(output_data_new, json_file, indent=2, ensure_ascii=False)

print(f"Output saved to {output_file_path_new}")

output_file_path_new = 'submission1.json'

with open(output_file_path_new, 'r', encoding='utf-8') as json_file:
    json_data = json.load(json_file)
print(json.dumps(json_data, indent=2)[:10000])
