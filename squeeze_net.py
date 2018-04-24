# -*- coding: utf-8 -*-


import tensorflow as tf
import tensorflow.contrib.layers as layers

"""
flags.DEFINE_integer('batch_size', 100, 'batch size')
flags.DEFINE_integer('num_epochs', 35, 'number of epochs')
flags.DEFINE_float('learning_rate', 0.04, 'init learning rate')
flags.DEFINE_float('dropout', 0.5, 'define dropout keep probability')
flags.DEFINE_float('max_grad_norm', 5.0, 'define maximum gradient normalize value')
flags.DEFINE_float('normalize_decay', 5.0, 'batch normalize decay rate')
flags.DEFINE_float('weight_decay', 0.0002, 'L2 regularizer weight decay rate')

flags.DEFINE_integer('print_every', 5, 'how often to print training status')
flags.DEFINE_string('name', None, 'name of result save dir')
"""

class SqueezeNet:
    def __init__(self, img_shape, num_classes, normalize_decay = 0.9, weight_decay = 0.0002, clip_norm = 5.0):
        self.num_classes = num_classes
        self.normalize_decay = normalize_decay
        self.weight_decay = weight_decay
        self.learning_rate = tf.placeholder(tf.float32)
        self.dropout = tf.placeholder(tf.float32)
        # batch data & labels
        self.train_data = tf.placeholder(tf.float32, shape=[None, img_shape[1], img_shape[2], img_shape[3]], name='train_data')
        # resize train image for squeeze net
        self.resized_data = self.train_data
        self.targets = tf.placeholder(tf.int32, shape=[None, 1], name='targets')
        self.is_training = tf.placeholder(tf.bool)
        logits = self.inference()
        # batch normalization in tensorflow requires this extra dependency
            
        self.loss = tf.reduce_mean(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.targets, logits=logits), name='loss')
        predictions = tf.argmax(tf.squeeze(logits, [1]), 1)
        correct_prediction = tf.equal(tf.cast(predictions, dtype=tf.int32), tf.squeeze(self.targets, [1]))
        self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

        optimizer = tf.train.AdamOptimizer(self.learning_rate)
        tvars = tf.trainable_variables() #权重集，相当于W和b
        self.global_step = tf.Variable(0, name='global_step', trainable=False)
        
        #Gradient Clipping的引入是为了处理gradient explosion或者gradients vanishing的问题。
        #Gradient Clipping的作用就是让权重（W或b）的更新限制在一个合适的范围。
        #Gradient Clipping计算所有权重梯度的平方和sumsq_diff，当scale_factor = clip_norm  / sumsq_diff <1 时将所有权重乘以scale_factor
        grads, global_norm = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), clip_norm)
        
        extra_update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(extra_update_ops):
            #将梯度应用在权重上
            self.train_op = optimizer.apply_gradients(zip(grads, tvars), global_step=self.global_step)
        
    def inference(self, scope='squeeze_net'):  # inference squeeze net
        with tf.variable_scope(scope):
            net = self.__conv2d(self.resized_data, 96, [3, 3], scope='conv_1') #[30, 30, 96]
            net = layers.max_pool2d(net, [3, 3], scope='max_pool_1')
            net = self._fire_module(net, 16, 64, scope='fire_2')
            net = self._fire_module(net, 16, 64, scope='fire_3')
            net = self._fire_module(net, 32, 128, scope='fire_4')
            net = layers.max_pool2d(net, [3, 3], scope='max_pool_2')
            net = self._fire_module(net, 32, 128, scope='fire_5')
            net = self._fire_module(net, 48, 192, scope='fire_6')
            net = self._fire_module(net, 48, 192, scope='fire_7')
            net = self._fire_module(net, 64, 256, scope='fire_8')
            net = layers.max_pool2d(net, [3, 3], scope='max_pool_3')
            net = self._fire_module(net, 64, 256, scope='fire_9')
            net = layers.dropout(net, self.dropout)
            net = self.__conv2d(net, self.num_classes, [1, 1], scope='conv_10')
            net = layers.avg_pool2d(net, [3, 3], stride=1, scope='avg_pool_1')
            return tf.squeeze(net, [2], name='logits')
        
    def _fire_module(self, input_tensor, squeeze_depth, expand_depth, scope=None):
        with tf.variable_scope(scope):
            squeeze_tensor = self.__squeeze(input_tensor, squeeze_depth)
            expand_tensor = self.__expand(squeeze_tensor, expand_depth)
        return expand_tensor
    
    def __conv2d(self, input_tensor, num_outputs, kernel_size, stride=1, scope=None):
        #decay是指在求滑动平均时的衰减，就是前面的数据影响会小一点
        #fused是一个融合了几个操作的bn，比普通bn速度快
        return layers.conv2d(input_tensor, num_outputs, kernel_size, stride=stride, scope=scope,data_format="NHWC",
                      weights_regularizer=layers.l2_regularizer(self.weight_decay), normalizer_fn=layers.batch_norm,
                      normalizer_params={'is_training': self.is_training, "fused" : True, "decay" : self.normalize_decay})
    
    def __squeeze(self, input_tensor, squeeze_depth):
        return self.__conv2d(input_tensor, squeeze_depth, [1, 1], scope="squeeze")
                           
    def __expand(self, input_tensor, expand_depth):
        expand_1X1 = self.__conv2d(input_tensor, expand_depth, [1, 1],scope='expand_1X1')
        expand_3X3 = self.__conv2d(input_tensor, expand_depth, [3, 3], scope='expand_3X3')    
        return tf.concat([expand_1X1, expand_3X3], 3)   
