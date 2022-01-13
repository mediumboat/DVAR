import math
import numpy as np
from pathlib import Path
import pandas as pd

def intersect2d(a, b):
    return np.array([x for x in set(tuple(x) for x in a) & set(tuple(x) for x in b)])


def load_data(dataset="ml"):
    global n_user, n_item, n_cate, user2user_encoded, df, movie2movie_encoded, item_cate, data_path
    if dataset == "ml":
        print("___Reading rating data___")
        data_path = Path("../../data/movielen/ml-100k")
        n_user = 943
        n_item = 1682
        n_cate = 19
        rating_file = data_path / "u.data"
        df = pd.read_csv(rating_file, names=["userId", "movieId", "rating"], usecols=[0, 1, 2], delimiter="\t")
        user_ids = np.sort(df["userId"].unique().tolist())
        user2user_encoded = {x: i for i, x in enumerate(user_ids)}
        movie_ids = np.sort(df["movieId"].unique().tolist())
        movie2movie_encoded = {x: i for i, x in enumerate(movie_ids)}
        df["user"] = df["userId"].map(user2user_encoded)
        df["item"] = df["movieId"].map(movie2movie_encoded)
        df = df[['user', 'item']]
        df = df.sample(frac=1.0).reset_index(drop=True)
        print("___Reading category data___")
        col_n = ['movieId', 'movie title', 'release date', 'video release date',
                 'IMDb URL', 'unknown', 'Action', 'Adventure', 'Animation',
                 "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy',
                 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi',
                 'Thriller', 'War', 'Western']
        tag_df = pd.read_csv(data_path / "u.item", sep='|', encoding='latin-1', names=col_n)
        tag_df["movie"] = tag_df["movieId"].map(movie2movie_encoded)
        tag_df.drop(columns=['movieId', 'movie title', 'release date', 'video release date', 'IMDb URL'],
                      inplace=True)
        item_cate = np.array([0, 0])
        genres_df = pd.read_csv(data_path / "u.genre", sep='|', encoding='latin-1', names=["genre", "idx"])
        genre_ids = np.sort(genres_df["idx"].unique().tolist())
        genre2genre_encoded = {x: i for i, x in enumerate(genre_ids)}
        genres_df["gid"] = genres_df["idx"].map(genre2genre_encoded)
        for _, row in tag_df.iterrows():
            for _, g in genres_df.iterrows():
                g_name = g["genre"]
                if row[g_name] == 1:
                    new_row = np.array([row["movie"], g["gid"]])
                    item_cate = np.row_stack((item_cate, new_row))
        item_cate = item_cate[1:, :]
    print("___Splitting train/val/testing data___")
    data = df.to_numpy()
    train_user_idx = math.floor(0.7 * n_user)
    val_user_idx = math.floor(0.85 * n_user)
    user_list = np.arange(n_user)
    item_list = np.arange(n_item)
    user_list_random = user_list.copy()
    user_list_idx = np.random.permutation(n_user)
    user_list_ramdom = user_list_random[user_list_idx]
    train_users = user_list_ramdom[: train_user_idx]
    train_data = df[df["user"].isin(train_users)].to_numpy()
    val_users = user_list_ramdom[train_user_idx: val_user_idx]
    val_data = df[df["user"].isin(val_users)].to_numpy()
    test_users = user_list_ramdom[val_user_idx:]
    test_data = df[df["user"].isin(test_users)].to_numpy()
    split_idx = np.random.choice(len(val_data), math.floor(len(val_data) * 0.5), replace=False)
    train_data = np.row_stack((train_data, val_data[split_idx, :]))
    val_data = np.delete(val_data, split_idx, 0)
    split_idx = np.random.choice(len(test_data), math.floor(len(test_data) * 0.5), replace=False)
    train_data = np.row_stack((train_data, test_data[split_idx, :]))
    train_users = np.unique(train_data[:, 0])
    test_data = np.delete(test_data, split_idx, 0)
    item_freq = np.zeros((n_item, 2), dtype=int)
    item, freq = np.unique(train_data[:, 1], return_counts=True)
    item_freq[item, 0] = item
    item_freq[item, 1] = freq
    negative_train_data = train_data.copy()
    n_positive_train_data = len(train_data)
    n_positive_val_data = len(val_data)
    n_positive_test_data = len(test_data)
    for user in train_users:
        interacted_items = train_data[np.where(train_data[:, 0] == user)[0], 1]
        n_interacted_items = len(interacted_items)
        uninteracted_items = np.setdiff1d(item_list, interacted_items)
        uninteracted_items_freq = item_freq[np.intersect1d(item_freq[:, 0], uninteracted_items), :]
        uninteracted_items_freq = uninteracted_items_freq[np.argsort(-uninteracted_items_freq[:, 1], ), :]
        negative_items = np.random.choice(uninteracted_items_freq[:, 0], n_interacted_items, replace=False,
                                          p=uninteracted_items_freq[:, 1]/sum(uninteracted_items_freq[:, 1]))
        negative_train_data[np.where(negative_train_data[:, 0] == user)[0], 1] = negative_items
    train_data = np.row_stack((train_data, negative_train_data))
    train_label = np.row_stack((np.ones((n_positive_train_data, 1)), np.zeros((n_positive_train_data, 1))))

    negative_val_data = np.zeros((20 * len(val_users), 2))
    n_negative_val_data = len(negative_val_data)
    i = 0
    for user in val_users:
        interacted_items = data[np.where(data[:, 0] == user)[0], 1]
        uninteracted_items = np.setdiff1d(item_list, interacted_items)
        negative_items = np.random.choice(uninteracted_items, 20, replace=False)
        negative_val_data[i:i+20, 0] = user
        negative_val_data[i:i+20, 1] = negative_items
        i = i + 20
    val_data = np.row_stack((val_data, negative_val_data))
    val_label = np.row_stack((np.ones((n_positive_val_data, 1)), np.zeros((n_negative_val_data, 1))))

    negative_test_data = np.zeros((20 * len(test_users), 2))
    n_negative_test_data = len(negative_test_data)
    i = 0
    for user in test_users:
        interacted_items = data[np.where(data[:, 0] == user)[0], 1]
        uninteracted_items = np.setdiff1d(item_list, interacted_items)
        negative_items = np.random.choice(uninteracted_items, 20, replace=False)
        negative_test_data[i:i + 20, 0] = user
        negative_test_data[i:i + 20, 1] = negative_items
        i = i + 20
    test_data = np.row_stack((test_data, negative_test_data))
    test_label = np.row_stack((np.ones((n_positive_test_data, 1)), np.zeros((n_negative_test_data, 1))))
    print("___Outputting data into hard disk___")
    train_data[:, 1] = train_data[:, 1] + n_user
    val_data[:, 1] = val_data[:, 1] + n_user
    test_data[:, 1] = test_data[:, 1] + n_user
    item_cate[:, 0] = item_cate[:, 0] + n_user
    item_cate[:, 1] = item_cate[:, 1] + n_user + n_item
    user_list = np.arange(n_user)
    item_list = np.arange(n_item) + n_user
    cate_list = np.arange(n_cate) + n_user + n_item
    with open(data_path/"user.node", 'wb+') as f1, open(data_path/"item.node", 'wb+') as f2, open(
            data_path/"cate.node", 'wb+') as f3:
        np.savetxt(f1, user_list, "%d")
        np.savetxt(f2, item_list, "%d")
        np.savetxt(f3, cate_list, "%d")
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
        f3.flush()
        f3.close()
    with open(data_path / "train.x", 'wb+') as f1, open(data_path / "train.y", 'wb+') as f2:
        np.savetxt(f1, train_data, "%d")
        np.savetxt(f2, train_label, "%d")
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
    with open(data_path / "val.x", 'wb+') as f1, open(data_path / "val.y", 'wb+') as f2:
        np.savetxt(f1, val_data, "%d")
        np.savetxt(f2, val_label, "%d")
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
    with open(data_path / "test.x", 'wb+') as f1, open(data_path / "test.y", 'wb+') as f2:
        np.savetxt(f1, test_data, "%d")
        np.savetxt(f2, test_label, "%d")
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
    with open(data_path / "item_cate.link", 'wb+') as f1:
        np.savetxt(f1, item_cate, "%d")
        f1.flush()
        f1.close()
    print("___Data loading is finished___")






if __name__ == "__main__":
    load_data()
