import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data ##NOTE change input data type
mnist = input_data.read_data_sets('MNIST_data', one_hot=True) #NOTE change data

"""
Creates layers

"""
def layer(input, weight_shape, bias_shape):
    """

    """
    weight_stddev = (2.0/weight_shape[0])**0.5 # weights should be square inverse # of inputs
    w_init = tf.random_normal_initializer(stddev=weight_stddev) #init with normal continous random integers
    bias_init = tf.constant_initializer(value = 0) # single value

    W = tf.get_variable("W", weight_shape, initializer=w_init) #
    b = tf.get_variable("b", bias_shape, initializer=bias_init)

    return tf.nn.relu(tf.matmul(input, W)+b) #get logit score returned

def inference(x, h1_w_shape, h1_b_shape, h2_w_shape,
                h2_b_shape, h3_w_shape, h3_b_shape, output_w_shape, output_b_shape):
    """
    Produces a probability distribution over the output classes given a mini-batch
    X is output layer count i.e. 0-9 for mnist means x=10

    Args:
        x: input tensor

        h1_w_shape: weight shape list [w,w-to-bias]
        h1_b_shape: bias list shape same as last weight shape [256] list

        h2_w_shape: weight shape list [784,256]
        h2_b_shape: bias list shape same as last weight shape [256] list

        output_w_shape: weight shape list [784,256]
        output_b_shape: bias list shape same as last weight shape [256] list

    """

    with tf.variable_scope("hidden_1"):
        hidden_1 = layer(x, h1_w_shape, h1_b_shape) # Creates hidden layer

    with tf.variable_scope("hidden_2"):
        hidden_2 = layer(hidden_1, h2_w_shape, h2_b_shape) # Created second hidden layer connected to first as input

    with tf.variable_scope("hidden_3"):
        hidden_3 = layer(hidden_2, h3_w_shape, h3_b_shape) # Created second hidden layer connected to first as input

    with tf.variable_scope("output"):
        output = layer(hidden_3, output_w_shape, output_b_shape) # connects h 2 with output by giving hidden 2 as input and 256 is weights to output layer 10

    return output

def loss(output, y):
    """
    Compute the average error per a data sample of mini-batch runs

    """
    xentropy = tf.nn.softmax_cross_entropy_with_logits(output, y)
    loss = tf.reduce_mean(xentropy) # computes the mean of all elements in tensor

    return loss

def training(cost, global_step):
    """
    Responsible for computing the gradients of the model's paramters and updating the model

    """

    tf.scalar_summary("cost", cost)

    optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
    train_op = optimizer.minimize(cost, global_step=global_step)

#   optimizer = tf.train.AdamOptimizer(learning_rate=0.0001, beta1=0.9, beta2=0.999, epsilon=1e-9, use_locking=False, name='Adam')
#   optimizer = tf.train.MomentumOptimizer(learning_rate, momentum)
#   optimizer = tf.train.RMSPropOptimizer(learning_rate, decay=0.9, momentum=0.0, epsilon=1e-10, use_locking=False, name='RMSProp')
#   optimizer = tf.train.AdagradOptimizer(learning_rate, initial_accumulator_value=0.1, use_locking=False, name='Adagrad')
#   optimizer = tf.train.GradientDescentOptimizer(learning_rate) #Creates the optimizer object
#   train_op = optimizer.minimize(cost, global_step=global_step) #Does training operation with cost and steps count

    return train_op

def evaluate(output, y):

    """
    Determine effectiveness of model
    """

    correct_prediction = tf.equal(tf.argmax(output, 1), tf.argmax(y, 1)) # deals with validation set here, 1 is the correct answer
    accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32)) # change prediction to float32
    return accuracy

#Hyper-parameters
learning_rate = 0.01
training_epochs = 400
batch_size = 100
display_step = 1
momentum = 0.9
#input weight shape and bias shape

h1ws = [784, 256]
h1bs = [256]

h2ws = [256, 256]
h2bs = [256]

h3ws = [256, 256]
h3bs = [256]

ows = [256, 10]
obs = [10]

with tf.Graph().as_default():

    """
    placehodlers for input and output
    """
    x = tf.placeholder("float", [None, 784]) # Holds the input matrix size 28x28
    y = tf.placeholder("float", [None, 10]) # Holds output product

    """
    Create tensor variables. Either restrictive
    """
    output = inference(x, h1ws, h1bs, h2ws, h2bs, h3ws, h3bs, ows, obs) #computes weights inputs and biases between layers
    cost = loss(output, y) #Calculates error after computing inference

    global_step = tf.Variable(0, name='global_step', trainable=False) #Var to hold step count

    train_op = training(cost, global_step) #Training operation now created with learning rate
    eval_op = evaluate(output, y) #Evaluation operation now prepared

    """
    Summary reporting objects
    """
    summary_op = tf.merge_all_summaries() #session op merging of summaries
    saver = tf.train.Saver()
    sess = tf.Session() #session create
    summary_writer = tf.train.SummaryWriter("Logistic_logs/", sess.graph) #session writing

    """
    Initialize all the variables and operations created above
    """
    init_op = tf.initialize_all_variables() #initalize all above

    """
    Start session for training
    """
    sess.run(init_op)

    #training cycle
    for epoch in range(training_epochs):

        avg_cost = 0

        total_batch = int(mnist.train.num_examples/batch_size) # Total number of possible batches

        #loop over the entire batch_size
        for i in range(total_batch):
            minibatch_x, minibatch_y = mnist.train.next_batch(batch_size) # Get raw number data and corresponding labels

            #fit the training using batch data
            feed_dict = {x: minibatch_x, y : minibatch_y} # raw data and labels into feed_dict tensor
            sess.run(train_op, feed_dict = feed_dict) # Feed the raw data and labels into train operation

            #Compute the average loss
            minibatch_cost = sess.run(cost, feed_dict=feed_dict) # Find error for latest training with data
            avg_cost += minibatch_cost/total_batch #Average out the total running cost

        #Display logs per epoch step
        if epoch % display_step == 0:

            val_feed_dict = {
                x : mnist.validation.images,
                y : mnist.validation.labels
            }
            accuracy = sess.run(eval_op, feed_dict=val_feed_dict)

            print "Validation Error: ", (1 - accuracy)

            summary_str = sess.run(summary_op, feed_dict=feed_dict)
            summary_writer.add_summary(summary_str, sess.run(global_step))

            saver.save(sess, "Logistic_logs/model-checkpoint", global_step=global_step)

        print "Optimization finished!"

        test_feed_dict = {
            x: mnist.test.images,
            y: mnist.test.labels
        }

        accuracy = sess.run(eval_op, feed_dict=test_feed_dict)

        print "Test Accuracy: ", accuracy
