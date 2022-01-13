from abc import ABC
import time
import numpy as np
import tensorflow as tf
from parameter import *
from pathlib import Path


class BaseEmbeddingModel(tf.keras.layers.Layer):
    def __init__(self, embeddings, dimension=DIMENSION):
        super(BaseEmbeddingModel, self).__init__()
        self.d = dimension
        self.node_embedding = embeddings
        self.W_k = self.add_weight(name="WK", shape=(self.d, self.d),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.W_q = self.add_weight(name="WQ", shape=(self.d, self.d),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.W_v = self.add_weight(name="WV", shape=(self.d, self.d),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.W_4 = self.add_weight(name="W4", shape=(self.d, 4 * self.d),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.b_4 = self.add_weight(name="b4", shape=(self.d, 1),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.W_3 = self.add_weight(name="W3", shape=(self.d, 3 * self.d),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))
        self.b_3 = self.add_weight(name="b3", shape=(self.d, 1),
                                   initializer='random_normal',
                                   trainable=True,
                                   regularizer=tf.keras.regularizers.L2(LAMBDA_THETA))

    def call(self, inputs, **kwargs):
        n = inputs.shape[0]
        l = inputs.shape[1]
        if l == 4:
            w_l = self.W_4
            b_l = self.b_4
        else:
            w_l = self.W_3
            b_l = self.b_3
        embeddings = tf.transpose(tf.gather(self.node_embedding, tf.cast(inputs, tf.int32)), [0, 2, 1])
        embeddings = tf.cast(embeddings, tf.float32)
        K_p = tf.matmul(self.W_k, embeddings)  # BATCH, d, l
        Q_p = tf.matmul(self.W_q, embeddings)  # BATCH, d, l
        V_p = tf.matmul(self.W_v, embeddings)  # BATCH, d, l
        S_p = tf.nn.softmax(tf.matmul(K_p, Q_p, transpose_a=True) / tf.sqrt(S))  # BATCH, l, l
        h_p = tf.nn.tanh(tf.matmul(w_l, tf.reshape(tf.matmul(V_p, S_p), [-1, DIMENSION * l, 1])) + b_l)
        h_m = tf.reduce_mean(h_p, axis=0)  # d, 1
        return h_m


class AdaptationModule(tf.keras.layers.Layer):
    def __init__(self, embeddings, item_cate):
        super(AdaptationModule, self).__init__()
        item_list = np.unique(item_cate[:, 0])
        self.embeddings = embeddings
        self.cate_embedding = embeddings.copy()
        for item in item_list:
            cate = item_cate[np.where(item_cate[:, 0] == item)[0], 1]
            self.cate_embedding[item, :] = np.mean(self.cate_embedding[cate, :], axis=0)
        self.cate_embedding = tf.cast(self.cate_embedding, tf.float32)
        self.embeddings = tf.cast(self.embeddings, tf.float32)
        self.alpha1 = self.add_weight(name="alpha_1", shape=(DIMENSION, DIMENSION), initializer='random_normal',
                                      trainable=True)
        self.beta1 = self.add_weight(name="beta_1", shape=(DIMENSION, DIMENSION), initializer='random_normal',
                                     trainable=True)
        self.alpha2 = self.add_weight(name="alpha_2", shape=(DIMENSION, DIMENSION), initializer='random_normal',
                                      trainable=True)
        self.beta2 = self.add_weight(name="beta_2", shape=(DIMENSION, DIMENSION), initializer='random_normal',
                                     trainable=True)

    def call(self, inputs, **kwargs):
        user, item = inputs
        user_e = tf.gather(self.embeddings, user)
        user_e = tf.reshape(user_e, [-1, DIMENSION, 1])
        alpha_u = tf.matmul(self.alpha1, user_e)  # BATCH, d, 1
        beta_u = tf.matmul(self.beta1, user_e)  # BATCH, d, 1
        alpha_u_i = tf.multiply((tf.matmul(self.alpha2, alpha_u) + beta_u),
                                tf.reshape(tf.gather(self.cate_embedding, item), [-1, DIMENSION, 1]))  # BATCH, d, 1
        beta_u_i = tf.multiply((tf.matmul(self.beta2, alpha_u) + beta_u),
                               tf.reshape(tf.gather(self.cate_embedding, item), [-1, DIMENSION, 1]))  # BATCH, d, 1
        return alpha_u_i, beta_u_i


class DVARModel(tf.keras.Model, ABC):
    def __init__(self, embeddings, item_cate, meta_paths):
        super(DVARModel, self).__init__()
        self.embeddings = tf.cast(embeddings, tf.float32)
        self.BaseModel = BaseEmbeddingModel(embeddings)
        self.AdaptationModel = AdaptationModule(embeddings, item_cate)
        for i in range(6):
            n = meta_paths[i].shape[0]
            l = meta_paths[i].shape[1]
            random_idx = tf.random.uniform(shape=(META_PATH_BATCH,), minval=0, maxval=n - 1, dtype=tf.int32)
            if i == 0:
                self.UICI = tf.gather(meta_paths[i], random_idx)
            if i == 1:
                self.UUUI = tf.gather(meta_paths[i], random_idx)
            if i == 2:
                self.UIUI = tf.gather(meta_paths[i], random_idx)
            if i == 3:
                self.UIII = tf.gather(meta_paths[i], random_idx)
            if i == 4:
                self.UII = tf.gather(meta_paths[i], random_idx)
            if i == 5:
                self.UUI = tf.gather(meta_paths[i], random_idx)
        self.meta_theta = self.add_weight(name="meta_theta", shape=(DIMENSION, 1),
                                   initializer='random_normal',
                                   trainable=True)
        self.mete_bias = self.add_weight(name="meta_bias", shape=(1, 1),
                                   initializer='random_normal',
                                   trainable=True)
        self.dense1 = tf.keras.layers.Dense(DIMENSION)
        self.dense2 = tf.keras.layers.Dense(DIMENSION)
        self.meta_path_embdding = None

    def call(self, inputs, training=None, mask=None):
        user, item, path = inputs
        h_uici = self.BaseModel(self.UICI)
        h_uuui = self.BaseModel(self.UUUI)
        h_uiui = self.BaseModel(self.UIUI)
        h_uiii = self.BaseModel(self.UIII)
        h_uii = self.BaseModel(self.UII)
        h_uui = self.BaseModel(self.UUI)
        self.meta_path_embdding = tf.transpose(tf.concat((h_uici, h_uuui, h_uiui, h_uiii, h_uii, h_uui), axis=1))
        path_emb = tf.gather(self.meta_path_embdding, path)
        self.alpha_u_i, self.beta_u_i = self.AdaptationModel((user, item))
        user_emb = tf.gather(self.embeddings, user)
        item_emb = tf.gather(self.embeddings, item)
        joint_emb = user_emb + path_emb - item_emb
        joint_emb = tf.reshape(joint_emb, [-1, 1, DIMENSION])
        joint_emb = self.dense1(joint_emb)
        joint_emb = self.dense2(joint_emb)
        w_theta = tf.multiply(self.alpha_u_i, self.meta_theta) + self.beta_u_i
        result = tf.nn.sigmoid(tf.matmul(w_theta, joint_emb, transpose_a=True, transpose_b=True) + self.mete_bias)
        return tf.squeeze(result)

    def compile(self,
              optimizer='rmsprop',
              loss=None,
              metrics=None,
              loss_weights=None,
              weighted_metrics=None,
              run_eagerly=None,
              **kwargs):
        super(DVARModel, self).compile()
        self.optimizer = optimizer
        self.loss = loss

    def train_step(self, data):
        x, y = data
        user = x[:, 0]
        item = x[:, 1]
        path = x[:, 2]
        with tf.GradientTape() as tape:
            y_predict = self.call((user, item, path))
            loss = self.loss(y, y_predict) + \
                   LAMBDA_FILM * (tf.nn.l2_loss(self.alpha_u_i-1) + tf.nn.l2_loss(self.beta_u_i))
        grads = tape.gradient(loss, self.trainable_variables)
        self.optimizer.apply_gradients(zip(grads, self.trainable_variables))
        return {"training_loss": loss}




if __name__ == "__main__":
    pass

