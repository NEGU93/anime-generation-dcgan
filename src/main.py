from model import DCGAN
import tensorflow as tf
import numpy as np
import os


def main(_):
    # Define Variables
    train = True
    epoch = 25
    learning_rate = 0.0002
    beta1 = 0.5
    train_size = np.inf
    # gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
    run_config = tf.ConfigProto()
    run_config.gpu_options.allow_growth = True
    sess = tf.Session(config=run_config)
    input_width = 108               # The size of image to use (will be center cropped). [108]
    input_height = 108              # The size of image to use (will be center cropped). [108]
    output_width = 64               # The size of the output images to produce [64]
    output_height = 64              # The size of the output images to produce [64]
    batch_size = 64                 # The size of batch images [64]
    sample_num = batch_size
    generate_test_images = 100      # Number of images to generate during test. [100]
    dataset = "celebA"              # "anime-faces"         # Name of the dataset
    input_frame_pattern = "*.jpg"   # Glob pattern of filename of input images [*]
    crop = True                     # TODO: ver cómo poner esto
    checkpoint_dir = "checkpoint"   # Directory name to save the checkpoints [checkpoint]
    sample_dir = "samples"          # Directory name to save the image samples [samples]
    data_dir = "./data"             # Root directory of dataset [data]

    if not os.path.exists(checkpoint_dir):
        os.makedirs(checkpoint_dir)
    if not os.path.exists(sample_dir):
        os.makedirs(sample_dir)

    # Initialize Convolutional GAN
    dcgan = DCGAN(
        sess,
        input_width=input_width,
        input_height=input_height,
        output_width=output_width,
        output_height=output_height,
        batch_size=batch_size,
        sample_num=sample_num,
        y_dim=10,
        z_dim=generate_test_images,
        dataset_name=dataset,
        input_fname_pattern=input_frame_pattern,
        crop=crop,
        checkpoint_dir=checkpoint_dir,
        sample_dir=sample_dir,
        data_dir=data_dir
    )

    if train:
        dcgan.train(learning_rate, beta1, epoch, data_dir, dataset, train_size, batch_size, checkpoint_dir)
    else:
        if not dcgan.load(checkpoint_dir)[0]:
            raise Exception("[!] Train a model first, then run test mode")


if __name__ == '__main__':
    tf.app.run()