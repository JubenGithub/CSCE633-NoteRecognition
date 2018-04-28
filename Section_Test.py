import tensorflow as tf
import sys
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import Section
import os
import glob
from scipy import signal, misc

#number of classes is 2 (squares and triangles)
nClass=32

#simple model (set to True) or convolutional neural network (set to False)
simpleModel=False

#dimensions of image (pixels)
height= 64
width= 64

#input is 2d Numpy array
#to show picture
def plot_image(image):
    plt.imshow(image, cmap = 'binary')
    plt.show()

#input: image any size, note type (black, black_rev, white,or white_rev)
#return: height position of input img    
def conv_noteMask(im, type_str):
    sum_height = 0
    count = 0
    
    for filename in glob.glob('./note_mask/'+ type_str +'/*.png'): #assuming gif
        mask=Image.open(filename).convert('L')
        mask = 1 - (np.array(mask, float)/255)
        rev_mask = np.flipud(np.fliplr(mask))
        res = signal.convolve2d(im, rev_mask, boundary='fill', mode='same')
        h,w = np.where(res == res.max()) #max value in
        if(len(h)!=1):
            h_val = int(h[0]) #more than 2 position has max value
        else:
            h_val = int(h)
        sum_height += h_val
        count += 1

    if count!=0 :
        return sum_height/count
    else:
        return 0

#input is 2d Numpy array().(1 x nClass)
# to output label meaning in word
def trans_label(label_arr):
    index = 0
    max_value = 0.0
    for i in range(nClass-1):
        if(label_arr[i] > max_value):
            index = i
            max_value = label_arr[i]
    return index

def shows_label(index):
    choices = {
    0: '2-2-Time',
    1: '2-4-Time',
    2: '3-4-Time',
    3: '3-8-Time',
    4: '4-4-Time',
    5: '6-8-Time',
    6: '9-8-Time',
    7: '12-8-Time',
    8: 'Barline',
    9: 'C-Clef',
    10: 'Common-Time',
    11: 'Cut-Time',
    12: 'Dot',
    13: 'Double-Sharp',
    14: 'Eighth-Note',
    15: 'Eighth-Rest',
    16: 'F-Clef',
    17: 'Flat',
    18: 'G-Clef',
    19: 'Half-Note',
    20: 'Natural',
    21: 'Quarter-Note',
    22: 'Quarter-Rest',
    23: 'Sharp',
    24: 'Sixteenth-Note',
    25: 'Sixteenth-Rest',
    26: 'Sixty-Four-Note',
    27: 'Sixty-Four-Rest',
    28: 'Thirty-Two-Note',
    29: 'Thirty-Two-Rest',
    30: 'Whole-Half-Rest',
    31: 'Whole-Note'}
    return choices.get(index,'None')

# main body below
tf.reset_default_graph()

# interactive session allows inteleaving of building and running steps
sess = tf.InteractiveSession()

# x is the input array, which will contain the data from an image
# this creates a placeholder for x, to be populated later
x = tf.placeholder(tf.float32, [None, width*height])
# similarly, we have a placeholder for true outputs (obtained from labels)
y_ = tf.placeholder(tf.float32, [None, nClass])

if simpleModel:
    # run simple model y=Wx+b given in TensorFlow "MNIST" tutorial
    print("Running Simple Model y=Wx+b")
    # initialise weights and biases to zero
    # W maps input to output so is of size: (number of pixels) * (Number of Classes)
    W = tf.Variable(tf.zeros([width*height, nClass]))
    # b is vector which has a size corresponding to number of classes
    b = tf.Variable(tf.zeros([nClass]))
    
    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    y = tf.nn.softmax(tf.matmul(x, W) + b)

else:
    # run convolutional neural network model given in "Expert MNIST" TensorFlow tutorial
    
    # functions to init small positive weights and biases
    def weight_variable(shape):
        initial = tf.truncated_normal(shape, stddev=0.1)
        return tf.Variable(initial)
    
    def bias_variable(shape):
        initial = tf.constant(0.1, shape=shape)
        return tf.Variable(initial)
    
    # set up "vanilla" versions of convolution and pooling
    def conv2d(x, W):
        return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
    def max_pool_2x2(x):
        return tf.nn.max_pool(x, ksize=[1, 2, 2, 1],
                              strides=[1, 2, 2, 1], padding='SAME')
    
    print ("Running Convolutional Neural Network Model")
    nFeatures1 = 16
    nFeatures2 = 36
    nFeatures3 = 64
    nFeatures4 = 128
    nNeuronsfc = 1024
    
    # use functions to init weights and biases
    # nFeatures1 features for each patch of size 5x5
    # SAME weights used for all patches
    # 1 input channel
    W_conv1 = weight_variable([5, 5, 1, nFeatures1])
    b_conv1 = bias_variable([nFeatures1])
    
    # reshape raw image data to 4D tensor. 2nd and 3rd indexes are W,H, fourth
    # means 1 colour channel per pixel
    # x_image = tf.reshape(x, [-1,28,28,1])
    x_image = tf.reshape(x, [-1,width,height,1])
    
    # hidden layer 1
    # pool(convolution(Wx)+b)
    # pool reduces each dim by factor of 2.
    h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
    h_pool1 = max_pool_2x2(h_conv1)
    
    # similarly for second layer, with nFeatures2 features per 5x5 patch
    # input is nFeatures1 (number of features output from previous layer)
    W_conv2 = weight_variable([5, 5, nFeatures1, nFeatures2])
    b_conv2 = bias_variable([nFeatures2])
    
    h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
    h_pool2 = max_pool_2x2(h_conv2)
    
    # similarly for second layer, with nFeatures3 features per 3x3 patch
    # input is nFeatures2 (number of features output from previous layer)
    W_conv3 = weight_variable([3, 3, nFeatures2, nFeatures3])
    b_conv3 = bias_variable([nFeatures3])
    
    h_conv3 = tf.nn.relu(conv2d(h_pool2, W_conv3) + b_conv3)
    h_pool3 = max_pool_2x2(h_conv3)
    
    W_conv4 = weight_variable([3, 3, nFeatures3, nFeatures4])
    b_conv4 = bias_variable([nFeatures4])
    
    h_conv4 = tf.nn.relu(conv2d(h_pool3, W_conv4) + b_conv4)
    h_pool4 = max_pool_2x2(h_conv4)
    
    
    # denseley connected layer. Similar to above, but operating
    # on entire image (rather than patch) which has been reduced by a factor of 4
    # in each dimension
    # so use large number of neurons
    
    # check our dimensions are a multiple of 4
    #if (width%4 or height%4):
    #  print "Error: width and height must be a multiple of 4"
    #  sys.exit(1)
    
    W_fc1 = weight_variable([(width//16) * (height//16) * nFeatures4, nNeuronsfc])
    b_fc1 = bias_variable([nNeuronsfc])
    
    # flatten output from previous layer
    h_pool4_flat = tf.reshape(h_pool4, [-1, (width//16) * (height//16) * nFeatures4])
    h_fc1 = tf.nn.relu(tf.matmul(h_pool4_flat, W_fc1) + b_fc1)
    
    # reduce overfitting by applying dropout
    # each neuron is kept with probability keep_prob
    keep_prob = tf.placeholder(tf.float32)
    h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
    
    # create readout layer which outputs to nClass categories
    W_fc2 = weight_variable([nNeuronsfc, nClass])
    b_fc2 = bias_variable([nClass])
    
    # define output calc (for each class) y = softmax(Wx+b)
    # softmax gives probability distribution across all classes
    # this is not run until later
    y=tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)


# measure of error of our model for both models
# this needs to be minimised by adjusting W and b
cross_entropy = tf.reduce_mean(-tf.reduce_sum(y_ * tf.log(y), reduction_indices=[1]))

# define training step which minimises cross entropy
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)

# argmax gives index of highest entry in vector (1st axis of 1D tensor)
correct_prediction = tf.equal(tf.argmax(y,1), tf.argmax(y_,1))

# get mean of all entries in correct prediction, the higher the better
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# initialize the variables
#sess.run(tf.global_variables_initializer())

# Add ops to save and restore all the variables.
saver = tf.train.Saver()

#load paths
ckpt_path = './model/617711/'
ckpt = tf.train.get_checkpoint_state(checkpoint_dir=ckpt_path)

saver.restore(sess, ckpt.model_checkpoint_path)
print('load done')

for i in range(0,400):
    filename = "WithoutStaff/"+str(i)+".png"
    ims = []
    ims = Section.crop_image(5, 3, filename)
    in_data_arr = np.zeros((len(ims), height*width))
    for index, im in enumerate(ims):
        in_data = np.array(im)
        in_data = 255 - in_data
        #print(in_data.shape)
        resize_in_data = in_data[:,:,0].reshape(1,height*width)
        in_data_arr[0,:] = resize_in_data
        dd_arr = np.zeros((1, nClass))
    #result = sess.run((tf.matmul(h_fc1_drop, W_fc2) + b_fc2), feed_dict={x: in_data_arr, y_: dd_arr, keep_prob: 1.0})
        result = sess.run(y, feed_dict={x: in_data_arr, y_: dd_arr, keep_prob: 1.0})
    #print(result)
        test_id = trans_label(result[0, :])
        test_result = shows_label(test_id)
        path_name = "Test_Result/"+str(test_result)
        file_name = "/"+str(i)+"-"+str(index)+".png"
        try:
            im.save(str(path_name+file_name))
        except IOError:
            os.mkdir(path_name)
            im.save(str(path_name+file_name))
        #ims[index].show()
#input()

#close the session to release resources
sess.close()
