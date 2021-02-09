import tensorflow as tf
import numpy as np
import scipy.sparse as sparse
import functools
import keras
from keras.layers import Layer
from keras import backend as K

def Mask(inputs, seq_len=20, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape)-2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def Dense(inputs, ouput_size, bias=True):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.nn.relu(outputs)
    #outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0))
    return outputs

def attention_fun(Q, K, scaled_=True,mask=False):
    attention = tf.multiply(Q, K)  # [batch_size, sequence_length, sequence_length]
    if scaled_:
        d_k = tf.cast(tf.shape(K)[-1], dtype=tf.float32)
        attention = tf.divide(attention, tf.sqrt(d_k))  # [batch_size, sequence_length, sequence_length]
    if mask == False:
        attention = Mask(attention)
    attention = tf.nn.softmax(attention, dim=-1)  # [batch_size, sequence_length, sequence_length]
    return attention

#使用矩阵分解搭建的网络结构
def inference_svd(user_batch,item_batch,user_num,item_num,dim=20,device="/cpu:0"):
    # 使用CPU
    training = tf.placeholder_with_default(False, shape=(), name = 'training')#给Batch norm加一个placeholder
    user_num = user_num+1
    item_num = item_num+1
    with tf.device("/cpu:0"):
        # 初始化几个bias
        my_batch_norm_layer = functools.partial(tf.layers.batch_normalization,
                              training=training, momentum=0.9)

        initializer_he = tf.keras.initializers.he_normal()
        global_bias=tf.get_variable("global_bias",shape=[])
        w_bias_user=tf.get_variable("embd_bias_user",shape=[user_num])
        w_bias_item=tf.get_variable("embd_bias_item",shape=[item_num])
        users = tf.placeholder(tf.int32, shape=[None, ], name='users')
        items = tf.placeholder(tf.int32, shape=[None, ], name='items')
        V = tf.Variable(tf.random_normal([10, 10], stddev=0.01))
        w0 = tf.Variable(tf.random_normal([10],-0.01,0.01))
        W = tf.Variable(tf.random_uniform([10,10],-0.01,0.01))
        # bias向量tf.random_uniform([input_size, ouput_size], -0.05, 0.05
        bias_user=tf.nn.embedding_lookup(w_bias_user,user_batch,name="bias_user")
        bias_item=tf.nn.embedding_lookup(w_bias_item,item_batch,name="bias_item")
        w_user=tf.get_variable("embd_user",shape=[user_num,dim],
                               initializer=initializer_he)
        w_item=tf.get_variable("embd_item",shape=[item_num,dim],
                               initializer=initializer_he)

        # user向量与item向量

        # user向量与item向量

        embd_user=tf.nn.embedding_lookup(w_user,user_batch,name="embedding_user")
        embd_item=tf.nn.embedding_lookup(w_item,item_batch,name="embedding_item")

        pr1 = np.dot(embd_user,embd_item)
        # 以上部分是TensorFlow初始化的部分 ，不管运行什么都需要的过程----------------
        user_user = tf.keras.layers.concatenate([embd_user,embd_user]) 
        item_item = tf.keras.layers.concatenate([embd_item,embd_item])

        pr3 = tf.layers.dense(user_user,100)
        pr3 = my_batch_norm_layer(pr3)
        pr3 = tf.nn.relu(pr3)
        pr3 = tf.layers.dense(pr3,100)
        pr3 = my_batch_norm_layer(pr3)
        pr3 = tf.nn.relu(pr3)
        pr3 = tf.layers.dense(pr3,10)
        pr3 = tf.layers.batch_normalization(pr3,training=training,momentum=0.9)
        pr3 = tf.nn.relu(pr3)


        pr4 = tf.layers.dense(item_item,100)
        pr4 = my_batch_norm_layer(pr4)
        pr4 = tf.nn.relu(pr4)
        pr4 = tf.layers.dense(pr4,100)
        pr4 = my_batch_norm_layer(pr4)
        pr4 = tf.nn.relu(pr4)
        pr4 = tf.layers.dense(pr4,10)
        pr4 = tf.layers.batch_normalization(pr4,training=training,momentum=0.9)
        pr4 = tf.nn.relu(pr4)
        pr2 = np.dot(pr3,pr4)
        pr5 = tf.keras.layers.concatenate([user_user,item_item])

        pr5 = tf.layers.dense(item_item,100)
        pr5 = my_batch_norm_layer(pr5)
        pr5 = tf.nn.relu(pr5)
        pr5 = tf.layers.dense(pr5,100)
        pr5 = my_batch_norm_layer(pr5)
        pr5 = tf.nn.relu(pr5)
        pr5 = tf.layers.dense(pr5,10)
        pr5 = tf.layers.batch_normalization(pr5,training=training,momentum=0.9)



        # bias_user and bias_item embedding
       
        #  self interactive attention
        '''
        embd_user_att = attention_fun(embd_user,embd_item)
        embd_item_att = attention_fun(embd_item,embd_user)

        #embd_user_att = Dense(embd_user_att,10)
        #embd_item_att = Dense(embd_item_att,10)
        embd_user_att = tf.layers.dense(embd_user_att,100,kernel_initializer=tf.contrib.layers.xavier_initializer())
        embd_user_att = my_batch_norm_layer(embd_user_att)
        embd_user_att = tf.nn.relu(embd_user_att)

        embd_item_att = tf.layers.dense(embd_item_att,100,kernel_initializer=tf.contrib.layers.xavier_initializer())
        embd_item_att = my_batch_norm_layer(embd_item_att)
        embd_item_att = tf.nn.relu(embd_item_att)

        embd_user_user_att = attention_fun(embd_user,embd_user)
        embd_item_item_att = attention_fun(embd_item,embd_item)

        embd_user_user_att = tf.layers.dense(embd_user_user_att,20,kernel_initializer=tf.contrib.layers.xavier_initializer(),)
        embd_user_user_att = my_batch_norm_layer(embd_user_user_att)
        embd_user_user_att = tf.nn.relu(embd_user_user_att)

        embd_item_item_att = tf.layers.dense(embd_item_item_att,20,kernel_initializer=tf.contrib.layers.xavier_initializer())
        embd_item_item_att = my_batch_norm_layer(embd_item_item_att)
        embd_item_item_att = tf.nn.relu(embd_item_item_att)

        pr2 = tf.keras.layers.dot([embd_user_user_att,embd_item_item_att],axes=1)
        pr3 = tf.keras.layers.concatenate([embd_user_user_att,embd_item_item_att],axis=1)
        pr3 = tf.layers.dense(pr3,100)
        pr3 = my_batch_norm_layer(pr3)
        pr3 = tf.nn.relu(pr3)
        pr3 = tf.layers.dense(pr3,100)
        pr3 = my_batch_norm_layer(pr3)
        pr3 = tf.nn.relu(pr3)
        pr3 = tf.layers.dense(pr3,10)
        pr3 = tf.layers.batch_normalization(pr3,training=training,momentum=0.9)
        pr3 = tf.nn.relu(pr3)

        pr4 = tf.multiply(embd_user,embd_item)
        pr4 = tf.multiply(pr4,embd_user)
        pr4 = Dense(pr4,100)
        pr4 = my_batch_norm_layer(pr4)
        pr4 = Dense(pr4,100)
        pr4 = my_batch_norm_layer(pr4)
        pr4 = Dense(pr4,10)
        pr4 = my_batch_norm_layer(pr4)
        '''


        '''
        embd_item_item_att = Dense(embd_item_item_att,100)
        embd_item_item_att = Dense(embd_item_item_att,10)

        embd_user_user_att = Dense(embd_user_user_att,100)
        embd_user_user_att = Dense(embd_user_user_att,10)
        '''

        #pr2 = np.dot(embd_item_item_att,embd_user_user_att)

        '''
        #### MultiHeadAttention
        user_self = multiheadAttention(embd_user,embd_user,embd_item)
        user_self = Dense(user_self,10)
        item_self = multiheadAttention(embd_item,embd_item,embd_user)
        item_self = Dense(item_self,10)

       
        embd_user_att1 = (tf.multiply(0.5,tf.reduce_sum(tf.subtract(tf.pow( tf.matmul(embd_user_att, tf.transpose(V)), 2), 
        tf.matmul(tf.pow(embd_user_att, 2), tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True)))
        embd_user_att2 = tf.add(w0,tf.reduce_sum( tf.multiply(W, embd_user_att), 1, keep_dims=True))
        embd_item_att1 = (tf.multiply(0.5,tf.reduce_sum(tf.subtract(tf.pow( tf.matmul(embd_item_att, tf.transpose(V)), 2), 
        tf.matmul(tf.pow(embd_item_att, 2), tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True)))
        embd_item_att2 =  tf.add(w0,tf.reduce_sum( tf.multiply(W, embd_item_att), 1, keep_dims=True))

        embd_user_item_att = attention_fun(tf.multiply(embd_item,embd_user),tf.multiply(embd_user,embd_item))
        embd_user_item_att = Dense(embd_user_item_att,10)
        fm_user_item = (tf.multiply(0.5,tf.reduce_sum(tf.subtract(tf.pow( tf.matmul(embd_user_item_att, tf.transpose(V)), 2), 
        tf.matmul(tf.pow(embd_user_item_att, 2), tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True)))
        #self attention
        embd_user_user_att = attention_fun(embd_user,embd_user,mask=True)
        embd_user_user_att = Dense(embd_user_user_att,10)
        embd_user_user_att = (tf.multiply(0.5,tf.reduce_sum(tf.subtract(tf.pow( tf.matmul(embd_user_user_att, tf.transpose(V)), 2), 
        tf.matmul(tf.pow(embd_user_user_att, 2), tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True)))

        embd_item_item_att = attention_fun(embd_item,embd_item,mask=True)
        embd_item_item_att = Dense(embd_user_user_att,10)

        embd_item_item_att = (tf.multiply(0.5,tf.reduce_sum(tf.subtract(tf.pow( tf.matmul(embd_item_item_att, tf.transpose(V)), 2), 
        tf.matmul(tf.pow(embd_item_item_att, 2), tf.transpose(tf.pow(V, 2)))), 1, keep_dims=True)))

        user_item_att = Dense(tf.keras.layers.concatenate([embd_user_user_att,embd_item_item_att]),10)
        user_item_att1 = Dense(tf.multiply(embd_user_user_att,embd_item_item_att),10)
        '''

    with tf.device(device):
        # 按照实际的公式进行计算 先对user向量和item向量求内积
        #infer0 = tf.reduce_sum(tf.multiply(embd_user,embd_item),1)
        infer = tf.reduce_sum(pr1,1)
        infer2 = tf.reduce_sum(pr2,1)
        infer3 = tf.reduce_sum(pr5,1)
        infer5 = tf.reduce_sum(pr4,1)
        infer6 = tf.reduce_sum(pr3,1)
        '''
        infer1 = tf.reduce_sum(embd_item_att,1)
        infer2 = tf.reduce_sum(embd_user_att,1)
        infer3 = tf.reduce_sum(embd_user_user_att,1)
        infer4 = tf.reduce_sum(embd_item_item_att,1)
        infer5 = tf.reduce_sum(pr2,1)
        infer6 = tf.reduce_sum(pr3,1)
        infer7 = tf.reduce_sum(pr4,1)
        '''

# 加上几个偏置项
        infer = tf.add(infer,global_bias)
        infer = tf.add(infer,bias_user)
        '''
        infer = tf.add(infer,infer1)
        infer = tf.add(infer,infer2)
        infer = tf.add(infer,infer3)
        infer = tf.add(infer,infer4)
        infer = tf.add(infer,infer5)
        infer = tf.add(infer,infer6)
        #infer = tf.add(infer,infer7)
        '''
        infer = tf.add(infer,infer2)
        infer = tf.add(infer,infer3)
        infer = tf.add(infer,infer5)
        infer = tf.add(infer,infer6)

        infer=tf.add(infer,bias_item,name="svd_inference")
        #infer = tf.keras.layers.Lambda(lambda x:x+tf.keras.backend.constant(3.5817,dtype=tf.keras.backend.floatx()))(infer)
        # 加上正则项
        regularize=tf.add(tf.nn.l2_loss(embd_user),tf.nn.l2_loss(embd_item),name="svd_inference")
        #regularize2 = tf.multiply(embd_user,embd_item)
        #regularize = tf.add(regularize,loss_fm)
        #regularize = tf.add(regularize,regularize2)

    return infer,regularize

def optimization(infer,regularize,rate_batch,learning_rate=0.000000001,reg=0.01,device="/cpu:0"):
    global_step=tf.train.get_global_step()
    assert global_step is not None
    # 选择合适的optimization做优化
    with tf.device(device):

        #learning_rate = tf.train.exponential_decay(learning_rate, global_step, 10000, 0.98, staircase=False)
        cost_l2=tf.nn.l2_loss(tf.subtract(infer,rate_batch))
        penalty=tf.constant(reg,dtype=tf.float32,shape=[],name="l2")
        cost=tf.add(cost_l2,tf.multiply(regularize,penalty))
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            train_op=tf.train.AdamOptimizer(learning_rate).minimize(cost,global_step=global_step)
    return cost,train_op
