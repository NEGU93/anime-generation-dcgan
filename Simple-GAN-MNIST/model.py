import tensorflow as tf

class Model:
    
    def __init__(self, learning_rate):
        
        self.X = tf.placeholder(tf.float32, shape = [None, 784])
        
        self.discriminator_W = tf.Variable(tf.random_normal([784, 128], stddev = 0.1))
        self.discriminator_b = tf.Variable(tf.zeros([128]))
        self.discriminator_W_out = tf.Variable(tf.random_normal([128, 1], stddev = 0.1))
        self.discriminator_b_out = tf.Variable(tf.zeros([1]))
        
        backpropagate_discriminator = [self.discriminator_W, self.discriminator_b, self.discriminator_W_out, self.discriminator_b_out]
        
        self.Z = tf.placeholder(tf.float32, shape = [None, 100])
        
        self.generator_W = tf.Variable(tf.random_normal([100, 128], stddev = 0.1))
        self.generator_b = tf.Variable(tf.zeros([128]))
        self.generator_W_out = tf.Variable(tf.random_normal([128, 784], stddev = 0.1))
        self.generator_b_out = tf.Variable(tf.zeros([784]))
        
        backpropagate_generator = [self.generator_W, self.generator_W_out, self.generator_b, self.generator_b_out]
        
        def discriminator(z):
            
            discriminator_hidden1 = tf.nn.relu(tf.matmul(z, self.discriminator_W) + self.discriminator_b)
            discriminator_out = tf.matmul(discriminator_hidden1, self.discriminator_W_out) + self.discriminator_b_out
            
            return discriminator_out
        
        def generator(z):
            
            generator_hidden1 = tf.nn.relu(tf.matmul(z, self.generator_W) + self.generator_b)
            generator_out = tf.matmul(generator_hidden1, self.generator_W_out) + self.generator_b_out
            
            return tf.nn.sigmoid(generator_out)
        
        self.generator_sample = generator(self.Z)
        discriminator_real = discriminator(self.X)
        discriminator_fake = discriminator(self.generator_sample)
        
        discriminator_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_real, labels = tf.ones_like(discriminator_real)))
        discriminator_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.zeros_like(discriminator_fake)))
        
        self.discriminator_total_loss = discriminator_loss_real + discriminator_loss_fake
        
        self.generator_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits = discriminator_fake, labels = tf.ones_like(discriminator_fake)))
        
        self.optimizer_discriminator = tf.train.AdamOptimizer(learning_rate).minimize(self.discriminator_total_loss, var_list = backpropagate_discriminator)
        self.optimizer_generator = tf.train.AdamOptimizer(learning_rate).minimize(self.generator_loss, var_list = backpropagate_generator)
        
        
        
        