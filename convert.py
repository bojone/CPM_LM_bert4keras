#! -*- coding: utf-8 -*-
# 转换PyTorch版的CPM-Generate权重为Tensorflow版的ckpt格式
# https://github.com/TsinghuaAI/CPM-Generate
# pytorch 1.7.0 + tensorflow 1.14.0 + keras 2.3.1
# 参考了 https://github.com/qhduan/CPM-LM-TF2

import numpy as np
import torch
import tensorflow as tf
import keras.backend as K
from tqdm import tqdm

num_hidden_layers = 32
out_file = '/root/kg/bert/CPM_LM_2.6B_TF/model.ckpt'

m0 = torch.load('./model-v1/80000/mp_rank_00_model_states.pt', map_location='cpu')
m1 = torch.load('./model-v1/80000/mp_rank_01_model_states.pt', map_location='cpu')


def single_weight(name):
    return m0['module'][name].numpy()

def merged_weight(name, axis=0):
    return np.concatenate([m0['module'][name].numpy(), m1['module'][name].numpy()], axis=axis)


tf_weights = {}

tf_weights['gpt/embeddings/word_embeddings'] = merged_weight('word_embeddings.weight')
tf_weights['gpt/embeddings/position_embeddings'] = single_weight('position_embeddings.weight')

qkv = ['query', 'key', 'value']
for i in range(num_hidden_layers):
    prefix = 'gpt/transformer/layer_%d/' % i
    w = merged_weight('transformer.layers.%s.attention.query_key_value.weight' % i)
    w = np.transpose(w)
    ws = [
        [w[:, :1280], w[:, 1280 * 3: 1280 * 4]],
        [w[:, 1280:1280 * 2], w[:, 1280 * 4: 1280 * 5]],
        [w[:, 1280 * 2:1280 * 3], w[:, 1280 * 5: 1280 * 6]]
    ]
    ws = [np.concatenate(w, axis=1) for w in ws]
    for k, w in zip(qkv, ws):
        name = prefix + 'attention/self/%s/kernel' % k
        tf_weights[name] = w
    b = merged_weight('transformer.layers.%s.attention.query_key_value.bias' % i)
    bs = [
        [b[:1280], b[1280 * 3: 1280 * 4]],
        [b[1280:1280 * 2], b[1280 * 4: 1280 * 5]],
        [b[1280 * 2:1280 * 3], b[1280 * 5: 1280 * 6]]
    ]
    bs = [np.concatenate(b, axis=0) for b in bs]
    for k, b in zip(qkv, bs):
        name = prefix + 'attention/self/%s/bias' % k
        tf_weights[name] = b
    w = merged_weight('transformer.layers.%s.attention.dense.weight' % i, axis=1)
    w = np.transpose(w)
    name = prefix + 'attention/output/dense/kernel'
    tf_weights[name] = w
    b = single_weight('transformer.layers.%s.attention.dense.bias' % i)
    name = prefix + 'attention/output/dense/bias'
    tf_weights[name] = b
    w = single_weight('transformer.layers.%s.input_layernorm.weight' % i)
    name = prefix + 'attention/input/LayerNorm/gamma'
    tf_weights[name] = w
    b = single_weight('transformer.layers.%s.input_layernorm.bias' % i)
    name = prefix + 'attention/input/LayerNorm/beta'
    tf_weights[name] = b
    w = single_weight('transformer.layers.%s.post_attention_layernorm.weight' % i)
    name = prefix + 'input/LayerNorm/gamma'
    tf_weights[name] = w
    b = single_weight('transformer.layers.%s.post_attention_layernorm.bias' % i)
    name = prefix + 'input/LayerNorm/beta'
    tf_weights[name] = b
    w = merged_weight('transformer.layers.%s.mlp.dense_h_to_4h.weight' % i)
    w = np.transpose(w)
    name = prefix + 'intermediate/dense/kernel'
    tf_weights[name] = w
    b = merged_weight('transformer.layers.%s.mlp.dense_h_to_4h.bias' % i)
    name = prefix + 'intermediate/dense/bias'
    tf_weights[name] = b
    w = merged_weight('transformer.layers.%s.mlp.dense_4h_to_h.weight' % i, axis=1)
    w = np.transpose(w)
    name = prefix + 'output/dense/kernel'
    tf_weights[name] = w
    b = single_weight('transformer.layers.%s.mlp.dense_4h_to_h.bias' % i)
    name = prefix + 'output/dense/bias'
    tf_weights[name] = b

tf_weights['gpt/output/LayerNorm/gamma'] = single_weight('transformer.final_layernorm.weight')
tf_weights['gpt/output/LayerNorm/beta'] = single_weight('transformer.final_layernorm.bias')


with tf.Graph().as_default():
    pairs = []
    for name, value in tf_weights.items():
        var = K.variable(tf.zeros(value.shape), name=name)
        pairs.append((var, value))
    with tf.Session() as sess:
        for pair in tqdm(pairs):
            K.set_value(*pair)
        saver = tf.train.Saver()
        saver.save(sess, out_file)
