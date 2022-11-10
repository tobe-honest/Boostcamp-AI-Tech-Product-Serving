import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader, Dataset

def age_map(x: int) -> int:
    x = int(x)
    if x < 20:
        return 1
    elif x >= 20 and x < 30:
        return 2
    elif x >= 30 and x < 40:
        return 3
    elif x >= 40 and x < 50:
        return 4
    elif x >= 50 and x < 60:
        return 5
    else:
        return 6

def process_context_data(users, books, ratings1, ratings2):
    ratings = pd.concat([ratings1, ratings2]).reset_index(drop=True)

    # 인덱싱 처리된 데이터 조인
    context_df = ratings.merge(users, on='user_id', how='left').merge(books.iloc[:, 1:], on='isbn', how='left')
    train_df = ratings1.merge(users, on='user_id', how='left').merge(books.iloc[:, 1:], on='isbn', how='left')
    test_df = ratings2.merge(users, on='user_id', how='left').merge(books.iloc[:, 1:], on='isbn', how='left')

    # DROP
    train_df.drop(['language', 'summary', 'book_title', 'book_author', 'img_url', 'img_path', 'remove_country_code', 'category', 'category_high'], axis=1, inplace=True)
    test_df.drop(['language', 'summary', 'book_title', 'book_author', 'img_url', 'img_path', 'remove_country_code', 'category', 'rating', 'category_high'], axis=1, inplace=True)

    # 인덱싱 처리
    loc_city2idx = {v:k for k,v in enumerate(context_df['location_city'].unique())}
    loc_state2idx = {v:k for k,v in enumerate(context_df['location_state'].unique())}
    loc_country2idx = {v:k for k,v in enumerate(context_df['location_country'].unique())}
    
    train_df['age'] = train_df['age'].fillna(int(train_df['age'].mean()))
    train_df['location_city'] = train_df['location_city'].fillna(train_df['location_city'].mode()[0])
    train_df['location_state'] = train_df['location_state'].fillna(train_df['location_state'].mode()[0])
    train_df['location_country'] = train_df['location_country'].fillna(train_df['location_country'].mode()[0])

    test_df['age'] = test_df['age'].fillna(int(test_df['age'].mean()))
    test_df['location_city'] = test_df['location_city'].fillna(test_df['location_city'].mode()[0])
    test_df['location_state'] = test_df['location_state'].fillna(test_df['location_state'].mode()[0])
    test_df['location_country'] = test_df['location_country'].fillna(test_df['location_country'].mode()[0])

    train_df['location_city'] = train_df['location_city'].map(loc_city2idx)
    train_df['location_state'] = train_df['location_state'].map(loc_state2idx)
    train_df['location_country'] = train_df['location_country'].map(loc_country2idx)
    test_df['location_city'] = test_df['location_city'].map(loc_city2idx)
    test_df['location_state'] = test_df['location_state'].map(loc_state2idx)
    test_df['location_country'] = test_df['location_country'].map(loc_country2idx)

    train_df['age'] = train_df['age'].apply(age_map)
    test_df['age'] = test_df['age'].apply(age_map)
    
    # book 파트 인덱싱
    publisher2idx = {v:k for k,v in enumerate(context_df['publisher'].unique())}
    language2idx = {v:k for k,v in enumerate(context_df['language'].unique())}
    author2idx = {v:k for k,v in enumerate(context_df['book_author'].unique())}

    year2idx = {v:k for k,v in enumerate(context_df['year_of_publication'].unique())}
    lang2idx = {v:k for k,v in enumerate(context_df['new_language'].unique())}
    english2idx = {v:k for k,v in enumerate(context_df['isenglish'].unique())}
    fiction2idx = {v:k for k,v in enumerate(context_df['isfiction'].unique())}

    train_df['publisher'] = train_df['publisher'].map(publisher2idx)
    train_df['new_language'] = train_df['new_language'].map(lang2idx)
    train_df['isenglish'] = train_df['isenglish'].map(english2idx)
    train_df['isfiction'] = train_df['isfiction'].map(fiction2idx)
    test_df['publisher'] = test_df['publisher'].map(publisher2idx)
    test_df['new_language'] = test_df['new_language'].map(lang2idx)
    test_df['isenglish'] = test_df['isenglish'].map(english2idx)
    test_df['isfiction'] = test_df['isfiction'].map(fiction2idx)

    train_df['year_of_publication'] = train_df['year_of_publication'].map(year2idx)
    test_df['year_of_publication'] = test_df['year_of_publication'].map(year2idx)

    idx = {
        "loc_city2idx":loc_city2idx,
        "loc_state2idx":loc_state2idx,
        "loc_country2idx":loc_country2idx,
        "publisher2idx":publisher2idx,
        "lang2idx":lang2idx,
        "english2idx":english2idx,
        "fiction2idx":fiction2idx,
        "year2idx":year2idx,
    }
    cols = ['user_id', 'isbn', 'age', 'location_city', 'location_state','location_country', 'publisher', 'new_language', 'isenglish', 'isfiction', 'year_of_publication'] + [str(i) for i in range(512)] + ['rating']
    train_df = train_df[cols]
    test_df = test_df[cols[:-1]]
    return idx, train_df, test_df


def context_data_load(args, l):

    ######################## DATA LOAD
    books = l[0]
    sub = l[1]
    test = l[2]
    train = l[3]
    users = l[4]
    ids = pd.concat([train['user_id'], sub['user_id']]).unique()
    isbns = pd.concat([train['isbn'], sub['isbn']]).unique()

    idx2user = {idx:id for idx, id in enumerate(ids)}
    idx2isbn = {idx:isbn for idx, isbn in enumerate(isbns)}

    user2idx = {id:idx for idx, id in idx2user.items()}
    isbn2idx = {isbn:idx for idx, isbn in idx2isbn.items()}

    train['user_id'] = train['user_id'].map(user2idx)
    sub['user_id'] = sub['user_id'].map(user2idx)
    test['user_id'] = test['user_id'].map(user2idx)
    users['user_id'] = users['user_id'].map(user2idx)

    train['isbn'] = train['isbn'].map(isbn2idx)
    sub['isbn'] = sub['isbn'].map(isbn2idx)
    test['isbn'] = test['isbn'].map(isbn2idx)
    books['isbn'] = books['isbn'].map(isbn2idx)

    idx, context_train, context_test = process_context_data(users, books, train, test)
    field_dims = np.array([len(user2idx), len(isbn2idx),
                            6, len(idx['loc_city2idx']), len(idx['loc_state2idx']), len(idx['loc_country2idx']),
                            len(idx['publisher2idx']), len(idx['lang2idx']), len(idx['english2idx']),
                            len(idx['fiction2idx']), len(idx['year2idx'])], dtype=np.uint32)
    data = {
            'train':context_train,
            'test':context_test,
            'field_dims':field_dims,
            'users':users,
            'books':books,
            'sub':sub,
            'idx2user':idx2user,
            'idx2isbn':idx2isbn,
            'user2idx':user2idx,
            'isbn2idx':isbn2idx,
            }


    return data


def context_data_split(args, data):
    X_train, X_valid, y_train, y_valid = train_test_split(
                                                        data['train'].drop(['rating'], axis=1),
                                                        data['train']['rating'],
                                                        test_size=args.TEST_SIZE,
                                                        random_state=args.SEED,
                                                        shuffle=True
                                                        )
    data['X_train'], data['X_valid'], data['y_train'], data['y_valid'] = X_train, X_valid, y_train, y_valid
    return data

def context_data_loader(args, data):

    train_dataset_fe = TensorDataset(torch.LongTensor(data['X_train'].iloc[:, :11].values), torch.FloatTensor(data['X_train'].iloc[:, 11:].values), torch.LongTensor(data['y_train'].values))
    valid_dataset_fe = TensorDataset(torch.LongTensor(data['X_valid'].iloc[:, :11].values), torch.FloatTensor(data['X_valid'].iloc[:, 11:].values), torch.LongTensor(data['y_valid'].values))
    test_dataset_fe = TensorDataset(torch.LongTensor(data['test'].iloc[:, :11].values), torch.FloatTensor(data['test'].iloc[:, 11:].values), )

    train_dataloader_fe = DataLoader(train_dataset_fe, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    valid_dataloader_fe = DataLoader(valid_dataset_fe, batch_size=args.BATCH_SIZE, shuffle=args.DATA_SHUFFLE)
    test_dataloader_fe = DataLoader(test_dataset_fe, batch_size=args.BATCH_SIZE, shuffle=False)

    data['train_dataloader'], data['valid_dataloader'], data['test_dataloader'] = train_dataloader_fe, valid_dataloader_fe, test_dataloader_fe

    return data
