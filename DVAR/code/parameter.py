import argparse
DATA_PATH = "../data/ml"
DIMENSION = 128
SVD_ITER = 20
FAISS_N_SIMILAR = 10 + 1
N_SIMILAR = 10
FAISS_N_LIST_I = 10
FAISS_N_LIST_U = 5
N_WALK_4 = 10000
N_WALK_3 = 10000

parser = argparse.ArgumentParser(description="Metapath2vec")
parser.add_argument('--path', type=str, default="../data/ml/meta.path" ,help="input_path")
parser.add_argument('--output_file', default="../data/ml/m2v.emb", type=str, help='output_file')
parser.add_argument('--dim', default=DIMENSION, type=int, help="embedding dimensions")
parser.add_argument('--window_size', default=7, type=int, help="context window size")
parser.add_argument('--iterations', default=5, type=int, help="iterations")
parser.add_argument('--batch_size', default=5000, type=int, help="batch size")
parser.add_argument('--care_type', default=0, type=int, help="if 1, heterogeneous negative sampling, else normal negative sampling")
parser.add_argument('--initial_lr', default=1e-3, type=float, help="learning rate")
parser.add_argument('--min_count', default=3, type=int, help="min count")
parser.add_argument('--num_workers', default=6, type=int, help="number of workers")
ARGS = parser.parse_args()

S = 128.0
LAMBDA_THETA = 0.01
LAMBDA_FILM = 0.01
META_PATH_BATCH = 5000
