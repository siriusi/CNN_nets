import tensorflow as tf
import numpy as np
import math
#import matplotlib.pyplot as plt
#%matplotlib inline
import time
import os
from mobile_net import MobileNet

import sys
from cs231n.data_utils import load_CIFAR10

def get_CIFAR10_data(num_training=49000, num_validation=1000, num_test=10000, cifar10_dir = None):
    """
    Load the CIFAR-10 dataset from disk and perform preprocessing to prepare
    it for the two-layer neural net classifier. These are the same steps as
    we used for the SVM, but condensed to a single function.  
    """
    # Load the raw CIFAR-10 data
    #cifar10_dir = 'cs231n/datasets'
    X_train, y_train, X_test, y_test = load_CIFAR10(cifar10_dir)

    # Subsample the data
    mask = range(num_training, num_training + num_validation)
    X_val = X_train[mask]
    y_val = y_train[mask]
    mask = range(num_training)
    X_train = X_train[mask]
    y_train = y_train[mask]
    mask = range(num_test)
    X_test = X_test[mask]
    y_test = y_test[mask]

    # Normalize the data: subtract the mean image
    mean_image = np.mean(X_train, axis=0)
    X_train -= mean_image
    X_val -= mean_image
    X_test -= mean_image

    return X_train, y_train, X_val, y_val, X_test, y_test



def run_model(session, Xd, yd, Xv, yv, epochs=3, batch_size=100,print_every=10, learning_rate = 0.04, dropout = 0.5):
    print("Batch dataset initialized.\n# of training data: {}\n# of test data: {}\n# of class: {}"
          .format(Xd.shape[0], Xv.shape[0], 10))
    
    # shuffle indicies
    train_indicies = np.arange(Xd.shape[0])
    np.random.shuffle(train_indicies)
    
    cnn_net = MobileNet(Xd.shape, 10)
        
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        # tensorboard setting
        train_summary = tf.summary.merge([tf.summary.scalar("train_loss", cnn_net.loss),
                                          tf.summary.scalar("train_accuracy", cnn_net.accuracy)])
        test_summary = tf.summary.merge([tf.summary.scalar("val_loss", cnn_net.loss),
                                         tf.summary.scalar("val_accuracy", cnn_net.accuracy)])
        
        fileName = time.strftime("%Y%m%d_%H%M%S", time.localtime())
        fileName = os.path.normcase("./result/" + fileName)
        print(fileName)
        summary_writer = tf.summary.FileWriter(fileName, sess.graph)
        yd = yd.reshape([Xd.shape[0], 1])
        yv = yv.reshape([Xv.shape[0], 1])
        for current_epoch in range(epochs):
            # training step
            ###for x_batch, y_batch in batch_set.batches():
            print("#############################Epoch Start##############################")
            
            for i in range(int(math.ceil(Xd.shape[0]/batch_size))):
                start = time.time()
                start_idx = (i*batch_size)%Xd.shape[0]
                idx = np.int32(train_indicies[start_idx:start_idx+batch_size])
                feed = {cnn_net.train_data:  Xd[idx,:, :, :], cnn_net.targets: yd[idx, :],
                        cnn_net.learning_rate: learning_rate, cnn_net.dropout: dropout, 
                        cnn_net.is_training : True}
                _, global_step, loss, accuracy, summary = \
                    sess.run([cnn_net.train_op, cnn_net.global_step, cnn_net.loss,
                              cnn_net.accuracy, train_summary], feed_dict=feed)
                summary_writer.add_summary(summary, global_step)
                if global_step % print_every == 0:
                    print("{}/{} ({} epochs) step, loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                          .format(global_step, int(round(Xd.shape[0]/batch_size)) * epochs, current_epoch,
                                  loss, accuracy, time.time() - start))
            #验证集
            start, avg_loss, avg_accuracy = time.time(), 0, 0

            feed = {cnn_net.train_data: Xv,cnn_net.targets: yv,
                    cnn_net.learning_rate: learning_rate, cnn_net.dropout: 1.0, cnn_net.is_training : False}
            loss, accuracy, summary = sess.run([cnn_net.loss, cnn_net.accuracy, test_summary], feed_dict=feed)
            avg_loss = loss
            avg_accuracy = accuracy 
            summary_writer.add_summary(summary, current_epoch)
            print("{} epochs test result. loss : {:.6f}, accuracy : {:.3f}, time/batch : {:.3f}sec"
                  .format(current_epoch, avg_loss , avg_accuracy , time.time() - start))
            print("\n")
    return avg_loss,avg_accuracy      


    
def main(_):
    if sys.platform == "linux" :
        cifar10_dir = "/home/z_tomcato/cs231n/assignment2/assignment2/cs231n/datasets/cifar-10-batches-py"
    else:
        cifar10_dir = 'cs231n/datasets'
    print("========================" + time.strftime("%Y%m%d_%H:%M:%S", time.localtime()) + "=========================")
    # Invoke the above function to get our data.
    X_train, y_train, X_val, y_val, X_test, y_test = get_CIFAR10_data(cifar10_dir = cifar10_dir)
    print('Train data shape: ', X_train.shape)
    print('Train labels shape: ', y_train.shape)
    print('Validation data shape: ', X_val.shape)
    print('Validation labels shape: ', y_val.shape)
    print('Test data shape: ', X_test.shape)
    print('Test labels shape: ', y_test.shape)
    
    tf.reset_default_graph()
    with tf.Session() as sess:
        #with tf.device("/cpu:0"): #"/cpu:0" or "/gpu:0" 
        sess.run(tf.global_variables_initializer())
        #print('Training')
        run_model(sess,X_train,y_train,X_val,y_val, epochs=4, batch_size=500,print_every=50, learning_rate = 0.005)
    print("==================================================================")
    print("\n")

if __name__ == '__main__':
    tf.app.run()