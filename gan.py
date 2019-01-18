import src
import keras.backend as K
import os
import numpy as np
import sys
import re
import math
import io
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from  matplotlib.animation import FuncAnimation
from matplotlib import colors
from netCDF4 import Dataset
from IPython.display import clear_output
#data folder
sys.path.insert(0, 'C:/Users/pkicsiny/Desktop/TUM/3/ADL4CV/data')
#forces CPU usage
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"   # see issue #152
os.environ["CUDA_VISIBLE_DEVICES"] = "0" #"" or "-1" for CPU, "0" for GPU
import tensorflow as tf
from tensorflow import keras
from keras.models import load_model
from tensorflow.python.client import device_lib
print(device_lib.list_local_devices())


# modified from source: https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
class GAN():
    def __init__(self,
                 dual=False,
                 past=1,
                 loss_function="l1",
                 augment=False,
                 g_dropout=0.5,
                 d_dropout=0.5,
                 g_batchnorm=True,
                 d_batchnorm=True,
                 obj=1,
                 bce_s=1,
                 bce_t=1,
                 dynamic_loss=False,
                 noisy_labels=False,
                 loss_constraint=0):

        self.dual = dual  # set this to True to train temporal discriminator
        self.size = 64
        self.g_batchnorm = g_batchnorm  # in G
        self.d_batchnorm = d_batchnorm  # in D
        self.g_dropout = g_dropout
        self.d_dropout = d_dropout
        self.noisy_labels = noisy_labels
        self.past_input = past  # set this to change sequence length
        self.input_shape = (self.size, self.size, self.past_input)  # 64, 64, t
        self.log = {"g_loss": [],
                    "d_loss": [],
                    "g_metric": [],
                    "d_metric": []}
        self.gradients = {"g_grads": [],
                          "ds_grads": [],
                          "dt_grads": []}
        self.inputs = []
        self.outputs = []
        self.train_data = None
        self.xval_data = None
        self.test_data = None
        self.augment = augment
        # Loss params
        self.loss_weights = [obj, bce_s]
        self.dynamic_loss = dynamic_loss
        self.objective_loss_constraint = loss_constraint
        # self.tenpercent_obj = obj*0.1
        self.losses = [src.custom_loss(loss=loss_function), keras.losses.binary_crossentropy]
        self.d_metric = [keras.metrics.binary_accuracy]
        # Optimizers
        self.d_optimizer = keras.optimizers.SGD(lr=0.01)
        # 0.01
        self.g_optimizer = keras.optimizers.Adam(0.0002, 0.5)  # , decay=1e-6)
        # lr=0.0002, 0.5


        # Build the generator
        self.generator = self.build_generator()
        # The generator takes a sequence of frames as input and generates the next image in that sequence
        input_img = keras.layers.Input(shape=self.input_shape)
        self.inputs.append(input_img)
        generated = self.generator(input_img)
        self.outputs.append(generated)

        # Build and compile spatial discriminator
        self.s_discriminator = self.build_discriminator("s")
        self.s_discriminator.compile(loss='binary_crossentropy',
                                     optimizer=self.d_optimizer,
                                     metrics=self.d_metric)
        # Spatial disc. takes the x as condition and G(x) and returns a float
        score_s = self.s_discriminator([input_img, generated])
        self.outputs.append(score_s)
        self.s_discriminator.trainable = False

        # Build and compile temporal discriminator (same as s disc. but has different inputs
        if self.dual:
            self.t_discriminator = self.build_discriminator("t")
            self.t_discriminator.compile(loss='binary_crossentropy',
                                         optimizer=self.d_optimizer,
                                         metrics=self.d_metric)
            # Temporal disc. takes in advected frame A(G(x_previous)) and G(x)
            adv = keras.layers.Input(shape=(64, 64, 1))
            self.inputs.append(adv)
            score_t = self.t_discriminator([adv, generated])
            self.outputs.append(score_t)
            self.t_discriminator.trainable = False
            self.losses.append(keras.losses.binary_crossentropy)
            self.loss_weights.append(bce_t)

        # Combined GAN model
        self.combined = keras.models.Model(inputs=self.inputs, outputs=self.outputs)
        # loss on all ouputs as a list: l1 loss on generated img, cross entropy for discriminator
        self.combined.compile(loss=self.losses, optimizer=self.g_optimizer, loss_weights=self.loss_weights)

    def build_generator(self, network="U-net"):
        generator = keras.Sequential()
        if network in ["Unet", "U-net", "unet", "u-net"]:
            return src.unet(self.input_shape, dropout=self.g_dropout, batchnorm=self.g_batchnorm)  # 64, 64, t

    def build_discriminator(self, which="s"):
        if which == "s":
            return src.spatial_discriminator(condition_shape=self.input_shape, dropout=self.d_dropout,
                                             batchnorm=self.d_batchnorm)
        elif which == "t":
            return src.temporal_discriminator(dropout=self.d_dropout, batchnorm=self.d_batchnorm)


            # ----------------------------------------------------
            #  Train
            # ----------------------------------------------------

    def train(self, epochs, d_epochs=1, batch_size=64, overfit=False):
        # Load the dataset
        if self.dual:
            self.past_input += 1
            print(
            "tempoGAN training: Increased input sequence length by one. First frame is only auxiliary for advection.")
        else:
            print("Normal GAN training: Changed dataset to 5min data.")

        print(f"Loading dataset.")
        self.train_data, self.xval_data, self.test_data = src.load_datasets(self.past_input)

        # split the dataset to inputs and ground truths
        gan_train, gan_truth, gan_val, gan_val_truth, gan_test, gan_test_truth = src.split_datasets(
            self.train_data[:300], self.xval_data, self.test_data, past_frames=self.past_input, augment=self.augment)

        vx, vy = src.optical_flow(gan_train[:, :, :, -2:-1], gan_train[:, :, :, -1:], window_size=4, tau=1e-2,
                                  init=1)  # (n,:,:,1)

        if overfit:
            batch_size = 1
            gan_train = gan_train[0:1]
            gan_truth = gan_truth[0:1]
            print("Overfit test: use only 1 sample.")

        # Adversarial ground truths
        real = np.ones((batch_size, 1))
        fake = np.zeros((batch_size, 1))
        print(f"Adversarial labels shape: {real.shape}")
        # Generator ground truths
        g_real = np.ones((batch_size, 1))
        # print(f"Initial loss weights: {self.loss_weights}")
        for epoch in range(epochs):

            # update loss weights
            if self.dynamic_loss and epoch > 0 and self.loss_weights[0] > 0.3:
                self.update_loss_weights(epoch)
            # ---------------------
            #  Train Discriminators
            # ---------------------

            # Train the first discriminator
            # inputs: [frame t, generated frame t+1 (from frame t)] & [frame t, ground truth of frame t (frame t+1)]
            # batches are unmixed
            self.s_discriminator.trainable = True
            for ks in range(d_epochs):
                # all 4D
                real_imgs, training_batch, generated_imgs, _, _ = self.create_training_batch(gan_train, gan_truth,
                                                                                             batch_size, "s")
                # print(f"Input shape: {training_batch.shape}\nGround truth shape: {real_imgs.shape} ")
                # mix 5% of labels
                if self.noisy_labels:
                    d_real, d_fake = self.noisy_d_labels(real, fake)
                    # print("Switching 5% of labels for spatial discriminator.")
                else:
                    d_real = real
                    d_fake = fake

                ds_loss_real = self.s_discriminator.train_on_batch([training_batch, real_imgs], d_real)
                ds_loss_fake = self.s_discriminator.train_on_batch([training_batch, generated_imgs], d_fake)
                ds_loss = 0.5 * np.add(ds_loss_real, ds_loss_fake)
                if d_epochs > 1:
                    print(f"    {ks} [Ds loss: {ds_loss[0]}, acc.: {100*ds_loss[1]}]")
            self.s_discriminator.trainable = False
            d_loss = ds_loss

            # true_xval = self.s_discriminator.predict([gan_val[:batch_size], gan_val_truth[:batch_size]])
            # fake_xval = self.generator.predict(gan_val[:batch_size])
            # fake_xval = self.s_discriminator.predict([gan_val[:batch_size], fake_xval])

            # Train the second discriminator
            # inputs: [advected generated frame t (from frame t-1), generated frame t+1 (from frame t)] &
            #        [advected ground truth of frame t-1 (advected frame t), ground truth frame t (frame t+1)]
            # batches are unmixed
            if self.dual:
                self.t_discriminator.trainable = True
                for kt in range(d_epochs):
                    real_imgs, training_batch, generated_imgs, advected_aux_gen, advected_aux_truth = self.create_training_batch(
                        gan_train, gan_truth, batch_size, "t",
                        vx, vy)
                    # only need rain map from the synthetics
                    if self.noisy_labels:
                        d_real, d_fake = self.noisy_d_labels(real, fake)
                        # print("Switching 5% of labels for spatial discriminator.")
                    else:
                        d_real = real
                        d_fake = fake

                    dt_loss_real = self.t_discriminator.train_on_batch([advected_aux_truth, real_imgs], d_real)
                    dt_loss_fake = self.t_discriminator.train_on_batch([advected_aux_gen, generated_imgs], d_fake)
                    dt_loss = 0.5 * np.add(dt_loss_real, dt_loss_fake)
                    # self.gradients["dt_grads"].append(self.get_gradients(self.t_discriminator))
                    if d_epochs > 1:
                        print(f"    {kt} [Dt loss: {dt_loss[0]}, acc.: {100*dt_loss[1]}]")
                d_loss = ds_loss + dt_loss
                self.t_discriminator.trainable = False

            # ---------------------
            #  Train Generator
            # ---------------------

            idx = np.random.randint(0, gan_train.shape[0], batch_size)
            if self.dual:
                training_truth = gan_truth[idx]  # frame t+1, 4D: n, 64, 64, 1
                aux_batch = gan_train[idx, :, :, :-1]  # from 0 to frame t-1 (not the last frame) 4D: n, 64, 64, past-1
                training_batch = gan_train[idx, :, :, 1:]  # from frame 1 to frame t, 4D: n, 64, 64, past-1
                aux_gen_imgs = self.generator.predict(aux_batch)  # 4D, n, 64, 64, 1: output is frame t
                # calculate optical flow for frame t-1 -> t
                # print("Calculating optical flow with Lucas Kanade method.")
                # vx, vy = src.optical_flow(aux_batch[:,:,:,-1:], training_batch[:,:,:,-1:], window_size=4, tau=1e-2)
                # concat channels
                aux_gen_imgs = np.concatenate((aux_gen_imgs, vx[idx], vy[idx]), axis=-1)  # n, 64, 64, 3
                # advect generated frame t
                advected_aux_gen = np.array([src.advect(sample) for sample in aux_gen_imgs])  # 4D (n, 64, 64, 1)
            else:
                training_batch = gan_train[idx]  # frame t or all past frames, 4D
                training_truth = gan_truth[idx]  # frame t+1 or all future frames, 4D
                # print(f"Input shape: {training_batch.shape}\nGround truth shape: {training_truth.shape} ")

            # Train the generator (to have the discriminator label samples as real)
            if self.dual:
                g_loss = self.combined.train_on_batch([training_batch, advected_aux_gen],
                                                      [training_truth, g_real, g_real])
            else:
                g_loss = self.combined.train_on_batch(training_batch, [training_truth, g_real])

            # Plot the progress
            self.log["g_loss"].append(g_loss)
            self.log["d_loss"].append(d_loss[0])
            # self.log["g_metric"].append(g_loss[1])
            self.log["d_metric"].append([d_loss[1], ds_loss_fake[1], ds_loss_real[1]])
            print(f"\033[1m {epoch}\n[D loss: {d_loss[0]}, acc.: {100*d_loss[1]}]" +
                  f"\033[1m[G loss: {np.round(g_loss[0], 4)}, obj.: {np.round(g_loss[1], 4)}," +
                  f"bce.: {np.round(g_loss[2], 4)}]\033[0m")  # , xval fake.: {np.mean(fake_xval)}, "+
            # f"xval true: {np.mean(true_xval)}]\033[0m\n"+

            # self.gradients["g_grads"].append(self.get_gradients(self.combined))


            # If at save interval => save generated image samples
            if epoch in [int(x) for x in np.linspace(0, 1, 21) * epochs]:  # 20 figures
                self.sample_images(epoch, gan_test, gan_test_truth)

    def create_training_batch(self, gan_train, gan_truth, batch_size, disc, vx=None, vy=None):
        idx = np.random.randint(0, gan_truth.shape[0], batch_size)
        # Generate a batch of new images
        if self.dual:
            # 0,1->2
            real_imgs = gan_truth[idx]  # frame t+1, 4D: n, 64, 64, 1
            training_batch = gan_train[idx, :, :, 1:]  # from frame 1 to end (t>=2), 4D: n, 64, 64, past-1
            generated_imgs = self.generator.predict(training_batch)  # n, h, w, 1, rho (4 dimensional, last drops)
            if disc == "t":
                aux_batch = gan_train[idx, :, :, :-1]  # from 0 to frame t-1 (not the last frame) 4D: n, 64, 64, past-1
                aux_gen_imgs = self.generator.predict(aux_batch)  # 4D, n, 64, 64, 1: output is frame t
                # calculate optical flow for frame t-1 -> t
                # print("Calculating optical flow with Lucas Kanade method.")
                # vx, vy = src.optical_flow(aux_batch[:,:,:,-1:], training_batch[:,:,:,-1:], window_size=4, tau=1e-2)
                # concat channels
                aux_gen_imgs = np.concatenate((aux_gen_imgs, vx[idx], vy[idx]), axis=-1)  # n, 64, 64, 3
                aux_true_imgs = training_batch[:, :, :, -1:]  # n, 64, 64, 1, frame t with all channels
                aux_true_imgs = np.concatenate((aux_gen_imgs, vx[idx], vy[idx]), axis=-1)  # n, 64, 64, 3
                # advected frame t (frame t+1)
                advected_aux_gen = np.array(
                    [src.advect(sample) for sample in aux_gen_imgs])  # 4D (n, h, w, m) (m: rho, vx, vy)
                advected_aux_truth = np.array([src.advect(sample) for sample in aux_true_imgs])  # 4D
            else:
                advected_aux_gen = None
                advected_aux_truth = None

        else:  # 4D
            real_imgs = gan_truth[idx]  # 4D
            training_batch = gan_train[idx]  # 4D
            generated_imgs = self.generator.predict(training_batch)  # 4D
            advected_aux_gen = None
            advected_aux_truth = None
        return real_imgs, training_batch, generated_imgs, advected_aux_gen, advected_aux_truth  # all 4D

    def sample_images(self, epoch, gan_test, gan_test_truth):
        n = 5
        if self.dual:
            test_batch = gan_test[:n, :, :, 1:]  # frame 1 to t (0 is not used bc. its only used in advection), 4D
        else:
            test_batch = gan_test[:n]
        test_truth = gan_test_truth[:n]
        gen_imgs = self.generator.predict(test_batch)
        plot_range = self.past_input if not self.dual else self.past_input - 1
        fig, axs = plt.subplots(n, plot_range + 2, figsize=(16, 16))
        for i in range(n):
            vmax = np.max([np.max(test_batch[i]), np.max(test_truth[i])])
            for j in range(plot_range):
                im = axs[i, j].imshow(test_batch[i, :, :, j], vmax=vmax)
                axs[i, j].axis('off')
                src.colorbar(im)
                axs[i, j].set_title("Frame t" + str([-self.past_input + 1 + j if j < self.past_input - 1 else ""][0]))
            im2 = axs[i, -2].imshow(test_truth[i, :, :, 0], vmax=vmax)
            axs[i, -2].axis('off')
            src.colorbar(im2)
            axs[i, -2].set_title("Frame t+1")
            im3 = axs[i, -1].imshow(gen_imgs[i, :, :, 0], vmax=vmax)
            axs[i, -1].axis('off')
            src.colorbar(im3)
            axs[i, -1].set_title("Prediction t+1")
        fig.savefig("Plots/epoch %d.png" % epoch)
        plt.close()

    def noisy_d_labels(self, real, fake):
        # idea: https://arxiv.org/pdf/1606.03498.pdf
        batch_size = len(real)
        five_percent = int(0.05 * batch_size)
        idx = np.random.randint(0, batch_size, five_percent)
        d_real = np.ones_like(real) * 0.9
        d_fake = np.zeros_like(fake)
        d_real[idx] = 0
        d_fake[idx] = 0.9
        return d_real, d_fake

    def get_gradients(self, model):
        """Return the gradient of every trainable weight in model

        Parameters
        -----------
        model : a keras model instance

        First, find all tensors which are trainable in the model. Surprisingly,
        `model.trainable_weights` will return tensors for which
        trainable=False has been set on their layer (last time I checked), hence the extra check.
        Next, get the gradients of the loss with respect to the weights.

        """
        weights = [tensor for tensor in model.trainable_weights]
        optimizer = model.optimizer

        return optimizer.get_gradients(model.total_loss, weights)

    def update_loss_weights(self, epoch):
        if epoch == self.objective_loss_constraint:  # objective function constraint
            self.loss_weights[0] -= 0.1  # self.tenpercent_obj
            # self.loss_weights[1:] = [x+self.tenpercent_obj for x in self.loss_weights[1:]]
            self.objective_loss_constraint += 20
            print(f"***Updated loss weights: {self.loss_weights}***\nNew threshold: {self.objective_loss_constraint}")
            self.combined.compile(loss=self.losses, optimizer=self.g_optimizer, loss_weights=self.loss_weights)

#d dropout= 0.25
gan= GAN(dual=True,
         augment=True,
         past=4,
         g_dropout=0,
         d_dropout=0.25,
         g_batchnorm=True,
         d_batchnorm=True,
         obj=0.1,
         bce_s=1,
         dynamic_loss = False,
         noisy_labels=True,
         loss_constraint=0)

gan.train(epochs=100, d_epochs=1, batch_size=64)
