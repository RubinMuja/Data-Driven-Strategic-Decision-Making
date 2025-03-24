import json
import pandas as pd
import random

def load_users(path='data/users.json'):
    with open(path, 'r') as f:
        users = json.load(f)
    df = pd.DataFrame(users)

    if 'profile' in df.columns:
        profile_df = pd.json_normalize(df['profile'])
        df = pd.concat([df.drop(columns=['profile']), profile_df], axis=1)

    possible_labels = ['bot'] * 50 + ['human'] * 50
    random.shuffle(possible_labels)
    df['label'] = possible_labels


    return df
