import os.path
from tqdm import tqdm
import networkx as nx
import numpy as np
from scipy import sparse
from sklearn.decomposition import TruncatedSVD
import faiss
from pathlib import Path
from parameter import *
import csrgraph as cg
from metapath2vec import Metapath2VecTrainer


def prepare(data_path=Path(DATA_PATH)):
    print("___Loading data___",end="")
    user_list = np.loadtxt(data_path/"user.node", dtype=np.ushort)
    n_users = len(user_list)
    item_list = np.loadtxt(data_path/"item.node", dtype=np.ushort)
    n_items = len(item_list)
    cate_list = np.loadtxt(data_path/"cate.node", dtype=np.ushort)
    n_cates = len(cate_list)
    item_cate = np.loadtxt(data_path/"item_cate.link", dtype=np.ushort)
    train_x = np.loadtxt(data_path/"train.x", dtype=np.ushort)
    train_y = np.loadtxt(data_path / "train.y", dtype=np.ushort)
    train_data = train_x[np.where(train_y == 1)[0], :]
    print("Done.")
    print("___Constructing graph___")
    print("___Computing similarity___", end="")
    interaction_matrix_csr = sparse.csr_matrix(
        (np.ones(len(train_data), dtype=np.bool),
         (train_data[:, 0], train_data[:, 1] - n_users)),
        shape=(n_users, n_items))
    svd = TruncatedSVD(n_components=DIMENSION, n_iter=SVD_ITER)
    user_feature = svd.fit_transform(interaction_matrix_csr)
    del interaction_matrix_csr
    interaction_matrix_csr = sparse.csr_matrix(
        (np.ones(len(train_data), dtype=np.bool),
         (train_data[:, 1] - n_users, train_data[:, 0])),
        shape=(n_items, n_users))
    svd = TruncatedSVD(n_components=DIMENSION, n_iter=SVD_ITER)
    item_feature = svd.fit_transform(interaction_matrix_csr)
    del interaction_matrix_csr
    quantizer = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, FAISS_N_LIST_U)
    index.train(user_feature.astype(np.float32))
    index.add(user_feature.astype(np.float32))
    _, simi_user = index.search(user_feature.astype(np.float32), FAISS_N_SIMILAR)
    simi_user = simi_user[:, 1:]
    quantizer = faiss.IndexFlatL2(DIMENSION)
    index = faiss.IndexIVFFlat(quantizer, DIMENSION, FAISS_N_LIST_I)
    index.train(item_feature.astype(np.float32))
    index.add(item_feature.astype(np.float32))
    _, simi_item = index.search(item_feature.astype(np.float32), FAISS_N_SIMILAR)
    simi_item = simi_item[:, 1:]
    simi_item = simi_item + n_users

    user_user_left = np.repeat(user_list, N_SIMILAR).reshape((-1, 1))
    user_user_right = np.array(simi_user).flatten().reshape((-1, 1))
    user_user = np.row_stack((np.column_stack((user_user_left, user_user_right)),
                              np.column_stack((user_user_right, user_user_left))))
    item_item_left = np.repeat(np.arange(n_items) + n_users, N_SIMILAR).reshape((-1, 1))
    item_item_right = np.array(simi_item).flatten().reshape((-1, 1))
    item_item = np.row_stack((np.column_stack((item_item_left, item_item_right)),
                              np.column_stack((item_item_right, item_item_left))))
    user_user = np.unique(user_user, axis=0)
    item_item = np.unique(item_item, axis=0)
    graph = nx.Graph()
    graph.add_edges_from(train_data)
    graph.add_edges_from(user_user)
    graph.add_edges_from(item_item)
    graph.add_edges_from(item_cate)
    print("Done.")
    print("___Initializing node embeddings___")
    print("___Starting Random Walk___")
    graph_scr = cg.csrgraph(graph)
    walks = graph_scr.random_walks(walklen=4,
                           epochs=N_WALK_4,
                           start_nodes=user_list)
    four_meta_path = np.unique(walks[np.where(np.in1d(walks[:, 3], item_list))[0], :], axis=0)
    del walks
    UICI = four_meta_path[np.where(np.in1d(four_meta_path[:, 2], cate_list))[0], :] #0
    UUUI_and_UIUI = four_meta_path[np.where(np.in1d(four_meta_path[:, 2], user_list))[0], :]
    UUUI = UUUI_and_UIUI[np.where(np.in1d(UUUI_and_UIUI[:, 1], user_list))[0], :] #1
    UIUI = UUUI_and_UIUI[np.where(np.in1d(UUUI_and_UIUI[:, 1], item_list))[0], :] #2
    UIII = four_meta_path[np.where(np.in1d(four_meta_path[:, 2], item_list))[0], :] #3
    del UUUI_and_UIUI
    del four_meta_path
    walks = graph_scr.random_walks(walklen=3,
                           epochs=N_WALK_3,
                           start_nodes=user_list)
    three_meta_path = np.unique(walks[np.where(np.in1d(walks[:, 2], item_list))[0], :], axis=0)
    UII = three_meta_path[np.where(np.in1d(three_meta_path[:, 1], item_list))[0], :] #4
    UUI = three_meta_path[np.where(np.in1d(three_meta_path[:, 1], user_list))[0], :] #5
    del graph_scr
    del three_meta_path
    if os.path.exists(data_path/"meta.path"):
        print("___Cleaning old files___", end="")
        os.remove(data_path/"meta.path")
        print("Done.")
    with open(data_path/"meta.path", 'ab+') as f:
        np.savetxt(f, UICI, fmt="%d")
        np.savetxt(f, UUUI, fmt="%d")
        np.savetxt(f, UIUI, fmt="%d")
        np.savetxt(f, UIII, fmt="%d")
        np.savetxt(f, UII, fmt="%d")
        np.savetxt(f, UUI, fmt="%d")
        f.flush()
        f.close()
    np.savetxt(data_path / "UICI.path",UICI, fmt="%d")
    np.savetxt(data_path / "UUUI.path", UUUI, fmt="%d")
    np.savetxt(data_path / "UIUI.path", UIUI, fmt="%d")
    np.savetxt(data_path / "UIII.path", UIII, fmt="%d")
    np.savetxt(data_path / "UII.path", UII, fmt="%d")
    np.savetxt(data_path / "UUI.path", UUI, fmt="%d")
    m2v = Metapath2VecTrainer(ARGS)
    m2v.train()
    node_embeddings_m2v = np.loadtxt(data_path/"m2v.emb", skiprows=1)
    node_embeddings = np.random.random((n_users + n_items + n_cates, DIMENSION))
    node_embeddings[node_embeddings_m2v[:,0].astype(int),:] = node_embeddings_m2v[ : , 1:]
    for cate_id in cate_list:
        items = item_cate[np.where(item_cate[:, 1] == cate_id)[0], 0]
        if len(items) > 0:
            node_embeddings[cate_id, :] = np.mean(node_embeddings[items, :], axis=0)
        else:
            print("Warning: none of the items belong to category No. "+cate_id)
    with open(data_path/"emb.npy", 'wb') as f:
        np.save(f, node_embeddings)
    print("___Counting meta_paths between training nodes___")
    train_data_with_meta_path = np.array([0, 0, 0], dtype=np.ushort)
    train_y_with_meta_path = list()
    i = 0
    for train_i in tqdm(train_x):
        user = train_i[0]
        item = train_i[1]
        label = train_y[i]
        if len(np.intersect1d(np.where(UICI[:, 0]==user)[0], np.where(UICI[:, 3]==item)[0])) > 0:
            new_data = np.array([user, item, 0], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        if len(np.intersect1d(np.where(UUUI[:, 0]==user)[0], np.where(UUUI[:, 3]==item)[0])) > 0:
            new_data = np.array([user, item, 1], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        if len(np.intersect1d(np.where(UIUI[:, 0]==user)[0], np.where(UIUI[:, 3]==item)[0])) > 0:
            new_data = np.array([user, item, 2], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        if len(np.intersect1d(np.where(UIII[:, 0]==user)[0], np.where(UIII[:, 3]==item)[0])) > 0:
            new_data = np.array([user, item, 3], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        if len(np.intersect1d(np.where(UII[:, 0]==user)[0], np.where(UII[:, 2]==item)[0])) > 0:
            new_data = np.array([user, item, 4], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        if len(np.intersect1d(np.where(UUI[:, 0]==user)[0], np.where(UUI[:, 2]==item)[0])) > 0:
            new_data = np.array([user, item, 5], dtype=np.ushort)
            train_data_with_meta_path = np.row_stack((train_data_with_meta_path, new_data))
            train_y_with_meta_path.append(label)
        i = i + 1
    train_data_with_meta_path = train_data_with_meta_path[1:, :]
    train_y_with_meta_path = np.array(train_y_with_meta_path, dtype=bool).reshape((-1, 1))
    with open(data_path/"train_meta_path.x", "wb+") as f1, open(data_path/"train_meta_path.y", "wb+") as f2:
        np.savetxt(f1, train_data_with_meta_path, fmt="%d")
        np.savetxt(f2, train_y_with_meta_path, fmt="%d")
        f1.flush()
        f1.close()
        f2.flush()
        f2.close()
    print("Done.")





    def train(data_path=Path(DATA_PATH)):
        node_embeddings = np.load(data_path/"emb.npy")
        train_x = np.loadtxt(data_path / "train_meta_path.x", dtype=np.ushort)
        train_y = np.loadtxt(data_path / "train_meta_path.y", dtype=np.ushort)
        val_x = np.loadtxt(data_path / "val.x", dtype=np.ushort)
        val_y = np.loadtxt(data_path / "val.y", dtype=np.ushort)
        test_x = np.loadtxt(data_path / "test.x", dtype=np.ushort)
        test_y = np.loadtxt(data_path / "test.y", dtype=np.ushort)













if __name__ == "__main__":
    prepare()
