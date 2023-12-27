import csv
import os

def load_csv(path_data):
    # Set a larger field size limit to avoid the _csv.Error
    csv.field_size_limit(131072 * 10)

    # Load shared_task_posts data
    shared_task_columns = ['post_id', 'user_id', 'timestamp', 'subreddit', 'post_title', 'post_body']
    shared_task_file_path = os.path.join(path_data, "shared_task_posts.csv")
    shared_task_posts = []

    with open(shared_task_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=shared_task_columns)
        next(reader, None)

        for row in reader:
            shared_task_posts.append(row)

    # Load crowd_train data
    crowd_train_columns = ['user_id', 'label']
    crowd_train_file_path = os.path.join(path_data, "crowd_train.csv")
    crowd_train = []

    with open(crowd_train_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=crowd_train_columns)
        next(reader, None)

        for row in reader:
            crowd_train.append(row)

    # Load task_A_train data
    task_A_train_columns = ['post_id', 'user_id', 'subreddit']
    task_A_train_file_path = os.path.join(path_data, "task_A_train.posts.csv")
    task_A_train = []

    with open(task_A_train_file_path, 'r', newline='', encoding='utf-8') as csvfile:
        reader = csv.DictReader(csvfile, fieldnames=task_A_train_columns)
        next(reader, None)

        for row in reader:
            task_A_train.append(row)

    return {
        'shared_task_posts': shared_task_posts,
        'crowd_train': crowd_train,
        'task_A_train': task_A_train
    }


def count_words(content):
    d={}
    for word in content:
        d[word]=d.get(word,0)+1
    return  d
# Example usage:
path_data_set = os.getcwd()
path_data_set = os.path.join(path_data_set, 'umd_reddit_suicidewatch_dataset_v2')
path_data_set = os.path.join(path_data_set, 'crowd')
path_data_set = os.path.join(path_data_set, 'train')

data_mapping = load_csv(path_data_set)

# Access the loaded data using keys
shared_task_posts_data = data_mapping['shared_task_posts']
crowd_train_data = data_mapping['crowd_train']
task_A_train_data = data_mapping['task_A_train']

# Now you can use the loaded data as needed
print("Shared Task Posts:")
print(shared_task_posts_data[:3])

print("\nCrowd Train:")
print(crowd_train_data[:3])

print("\nTask A Train:")
print(task_A_train_data[:3])
print()
for task in task_A_train_data:
    for shared_task in shared_task_posts_data:
        if task['post_id']==shared_task['post_id']:
            # title=shared_task['post_title']
            # body=shared_task['post_body']
            # task['post_id']={'post_id':task['post_id'],'title':title,'body':body}
            task['post_id']=shared_task
            break
    for crowd in crowd_train_data:
        if task['user_id'] == crowd['user_id']:
            task['user_id'] = crowd
            break

print("\nTask A Train:")
print(task_A_train_data[:3])
print()