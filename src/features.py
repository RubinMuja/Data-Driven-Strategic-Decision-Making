def extract_features(df):
    df['description_length'] = df['description'].apply(lambda x: len(x) if isinstance(x, str) else 0)
    df['tweet_count'] = df['statuses_count']
    df['followers_count'] = df['followers_count']
    df['friends_count'] = df['friends_count']
    df['listed_count'] = df['listed_count']
    
    return df[[
        'description_length', 'tweet_count',
        'followers_count', 'friends_count', 'listed_count'
    ]].fillna(0)
