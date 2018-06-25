import model
import time
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
import tensorflow as tf


def sample_noise(m, n):
    return np.random.uniform(-1., 1., size = [m, n])

# i suggest to use multiply of 4
sample_size_output = 16

sample_size_train = 128
learning_rate = 0.001

epoch = 100000

mnist = input_data.read_data_sets('/home/project/imagerecognition')

sess = tf.InteractiveSession()
model = model.Model(learning_rate)
sess.run(tf.global_variables_initializer())

EPOCH = []; DISCRIMINATOR_LOSS = []; GENERATOR_LOSS = []

for i in range(0,epoch):
    
    EPOCH.append(i)
    
    last_time = time.time()
    input_images, _ = mnist.train.next_batch(sample_size_train)
    
    _, discriminator_loss = sess.run([model.optimizer_discriminator, model.discriminator_total_loss], feed_dict = {model.X : input_images, model.Z : sample_noise(sample_size_train, 100)})
    _, generator_loss = sess.run([model.optimizer_generator, model.generator_loss], feed_dict = {model.Z : sample_noise(sample_size_train, 100)})
    
    DISCRIMINATOR_LOSS.append(discriminator_loss); GENERATOR_LOSS.append(generator_loss)
    
    if (i + 1) % 1000 == 0:
        
        print("epoch: " + str(i + 1) + ", discriminator loss: " + str(discriminator_loss) + ", generator loss: " + str(generator_loss) + ", s / batch epoch: " + str(time.time() - last_time))
        
        fig = plt.figure(figsize = (4, 4))
        
        samples = sess.run(model.generator_sample, feed_dict = {model.Z: sample_noise(sample_size_output, 100)})
        
        for z in range(0,sample_size_output):
            
            plt.subplot(sample_size_output / 4, 4, z + 1)
            plt.imshow(samples[z].reshape(28, 28), cmap = 'Greys_r')
        
        plt.savefig('sample' + str((i + 1) / 1000) + '.png')
        plt.savefig('sample.pdf')
        plt.cla()
        
        import seaborn as sns
        sns.set()
        fig = plt.figure(figsize = (5, 5))
        plt.plot(EPOCH, DISCRIMINATOR_LOSS, label = 'discriminator loss')
        plt.plot(EPOCH, GENERATOR_LOSS, label = 'generator loss')
        plt.xlabel('epoch'); plt.ylabel('loss')
        plt.legend()
        plt.savefig('loss.png')
        plt.cla()
