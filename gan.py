# modified from source: https://github.com/eriklindernoren/Keras-GAN/blob/master/gan/gan.py
class GAN():
    def __init__(self, dual=False, past=1, loss_function="l1",
                 augment=False, g_dropout=0.5, d_dropout=0.5, batchnorm=True, obj=10, bce_s=0, bce_t=0):
        self.dual = dual  # set this to True to train temporal discriminator
        self.size = 64
        self.g_dropout = g_dropout
        self.d_dropout = d_dropout
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
        self.batchnorm = batchnorm
        # Loss params
        self.loss_weights = [obj, bce_s]
        self.losses = [src.custom_loss(loss=loss_function), keras.losses.binary_crossentropy]
        self.d_metric = [keras.metrics.binary_accuracy]

        self.d_optimizer = keras.optimizers.SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True)
        # 0.01
        self.g_optimizer = keras.optimizers.Adam(0.0002, 0.5, decay=1e-6)
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
            adv = keras.layers.Input(shape=self.input_shape)
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
            return src.unet(self.input_shape, dropout=self.g_dropout, batchnorm=self.batchnorm)  # 64, 64, t

    def build_discriminator(self, which="s"):
        if which == "s":
            return src.spatial_discriminator(condition_shape=self.input_shape, dropout=self.d_dropout)
        elif which == "t":
            return src.temporal_discriminator()
            # ---------------------
            #  Train
            # ---------------------

    def train(self, epochs, d_epochs=1, dataset="5min", batch_size=64, overfit=False):
        assert isinstance(d_epochs, int) > 0 and isinstance(epochs,
                                                            int) > 0, "Number of epochs must be a positive integer."

        # Load the dataset
        if self.dual:
            if dataset not in ["gan", "GAN", "tempogan", "tempoGAN"]:
                dataset = "gan"
                print("tempoGAN training: Changed dataset to GAN data.")
            self.past_input += 1
            print(
            "tempoGAN training: Increased input sequence length by one. First frame is only auxiliary for advection.")
        else:
            if dataset in ["gan", "GAN", "tempogan", "tempoGAN"]:
                dataset = "5min"
                print("Normal GAN training: Changed dataset to 5min data.")

        if overfit:
            batch_size = 1
            print("Overfit test: batch size set to 1.")

        print(f"Loading {dataset} dataset.")
        self.train_data, self.xval_data, self.test_data = src.load_datasets(dataset, self.past_input)
        self.train_data[np.isnan(self.train_data)] = 0
        self.xval_data[np.isnan(self.xval_data)] = 0
        self.test_data[np.isnan(self.test_data)] = 0
        # split the dataset to inputs and ground truths
        gan_train, gan_truth, gan_val, gan_val_truth, gan_test, gan_test_truth = src.split_datasets(
            self.train_data, self.xval_data, self.test_data, past_frames=self.past_input, augment=self.augment)

        if overfit:
            batch_size = 1
            gan_train = gan_train[0:1]
            gan_truth = gan_truth[0:1]
            print("Overfit test: batch size set to 1.")

        # Adversarial ground truths
        real = np.ones((batch_size, 1))  # *0.9
        fake = np.zeros((batch_size, 1))
        # Generator ground truths
        g_real = np.ones((batch_size, 1))
        print(f"Initial loss weights: {self.loss_weights}")
        for epoch in range(epochs):

            # update loss weights
            if epoch % (epochs / 10) == 0 and epoch > 0 and 3 * self.loss_weights[0] > self.loss_weights[1]:
                self.update_loss_weights()
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
                                                                                             batch_size)
                # mix 5% of labels
                d_real, d_fake = self.noisy_d_labels(real, fake)
                ds_loss_real = self.s_discriminator.train_on_batch([training_batch, real_imgs], d_real)
                ds_loss_fake = self.s_discriminator.train_on_batch([training_batch, generated_imgs], d_fake)
                ds_loss = 0.5 * np.add(ds_loss_real, ds_loss_fake)
                if d_epochs > 1:
                    print(f"    {ks} [Ds loss: {ds_loss[0]}, acc.: {100*ds_loss[1]}]")
            true_xval = self.s_discriminator.predict([gan_val[:batch_size], gan_val_truth[:batch_size]])
            fake_xval = self.generator.predict(gan_val[:batch_size])
            fake_xval = self.s_discriminator.predict([gan_val[:batch_size], fake_xval])
            d_loss = ds_loss
            self.s_discriminator.trainable = False

            # Train the second discriminator
            # inputs: [advected generated frame t (from frame t-1), generated frame t+1 (from frame t)] &
            #        [advected ground truth of frame t-1 (advected frame t), ground truth frame t (frame t+1)]
            # batches are unmixed
            if self.dual:
                self.t_discriminator.trainable = True
                for kt in range(d_epochs):
                    real_imgs, training_batch, generated_imgs, advected_aux_gen, advected_aux_truth = self.create_training_batch(
                        gan_train, gan_truth, batch_size)
                    # only need rain map from the synthetics
                    d_real, d_fake = self.noisy_d_labels(real, fake)
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
                training_truth = gan_truth[idx, :, :, :, 0]  # frame t+1, 4D: n, 64, 64, 1
                assert training_truth.shape[-1] == 1, f"real_imgs: (n, 64, 64, 1), {real_imgs.shape}"
                aux_batch = gan_train[idx, :, :, :-1,
                            0]  # from 0 to frame t-1 (not the last frame) 4D: n, 64, 64, past-1
                assert aux_batch.shape[
                           -1] == self.past_input - 1, f"aux_batch: (n, 64, 64, {self.past_input-1}), {aux_batch.shape}"
                training_batch = gan_train[idx, :, :, 1:, 0]  # from frame 1 to end (t>=2), 4D: n, 64, 64, past-1
                assert training_batch.shape[
                           -1] == self.past_input - 1, f"training_batch: (n, 64, 64, {self.past_input-1}), {training_batch.shape}"
                aux_gen_imgs = self.generator.predict(aux_batch)  # 4D, n, 64, 64, 1: output is frame t
                assert aux_gen_imgs.shape[-1] == 1, f"aux_gen_imgs: (n, 64, 64, 1), {aux_gen_imgs.shape}"
                # append velocity field of frame t (last instance of past sequence)
                aux_gen_imgs = np.concatenate((aux_gen_imgs, gan_train[idx, :, :, -1, 1:]), axis=-1)  # n, 64, 64, 3
                assert aux_gen_imgs.shape[-1] == 3, f"aux_gen_imgs: (n, 64, 64, 3), {aux_gen_imgs.shape}"
                # advect generated frame t
                advected_aux_gen = np.array(
                    [src.advect(sample) for sample in aux_gen_imgs])  # 4D (n, h, w, m) (m: rho, vx, vy)
                assert advected_aux_gen.shape[-1] == 1, f"advected_aux_gen: (n, 64, 64, 1), {advected_aux_gen.shape}"
            else:
                training_batch = gan_train[idx]  # frame t or all past frames, 4D
                training_truth = gan_truth[idx]  # frame t+1 or all future frames, 4D

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
            self.log["d_metric"].append(d_loss[1])
            print(f"\033[1m {epoch}\n[D loss: {d_loss[0]}, acc.: {100*d_loss[1]}, xval fake.: {np.mean(fake_xval)}, " +
                  f"xval true: {np.mean(true_xval)}]\033[0m\n" +
                  f"\033[1m[G loss: {np.round(g_loss[0], 4)}, obj.: {np.round(g_loss[1], 4)}," +
                  f"bce.: {np.round(g_loss[2], 4)}]\033[0m")
            # self.gradients["g_grads"].append(self.get_gradients(self.combined))


            # If at save interval => save generated image samples
            if epoch in [int(x) for x in np.linspace(0, 1, 11) * epochs]:
                self.sample_images(epoch, gan_test, gan_test_truth)

    def create_training_batch(self, gan_train, gan_truth, batch_size):
        idx = np.random.randint(0, gan_truth.shape[0], batch_size)
        # Generate a batch of new images
        if self.dual:
            # 0,1->2
            real_imgs = gan_truth[idx, :, :, :, 0]  # frame t+1, 4D: n, 64, 64, 1
            assert real_imgs.shape[-1] == 1, f"real_imgs: (n, 64, 64, 1), {real_imgs.shape}"
            training_batch = gan_train[idx, :, :, 1:, 0]  # from frame 1 to end (t>=2), 4D: n, 64, 64, past-1
            assert training_batch.shape[
                       -1] == self.past_input - 1, f"training_batch: (n, 64, 64, {self.past_input-1}), {training_batch.shape}"
            aux_batch = gan_train[idx, :, :, :-1, 0]  # from 0 to frame t-1 (not the last frame) 4D: n, 64, 64, past-1
            assert aux_batch.shape[
                       -1] == self.past_input - 1, f"aux_batch: (n, 64, 64, {self.past_input-1}), {aux_batch.shape}"
            generated_imgs = self.generator.predict(training_batch)  # n, h, w, 1, rho (4 dimensional, last drops)
            assert generated_imgs.shape[-1] == 1, f"generated_imgs: (n, 64, 64, 1), {generated_imgs.shape}"
            aux_gen_imgs = self.generator.predict(aux_batch)  # 4D, n, 64, 64, 1: output is frame t
            assert aux_gen_imgs.shape[-1] == 1, f"aux_gen_imgs: (n, 64, 64, 1), {aux_gen_imgs.shape}"
            # append velocity fields of frame t
            # this will be advected
            aux_gen_imgs = np.concatenate((aux_gen_imgs, gan_train[idx, :, :, -1, 1:]), axis=-1)  # n, 64, 64, 3
            assert aux_gen_imgs.shape[-1] == 3, f"aux_gen_imgs: (n, 64, 64, 3), {aux_gen_imgs.shape}"
            aux_true_imgs = gan_train[idx, :, :, -1]  # n, 64, 64, 3, frame t with all channels
            assert aux_true_imgs.shape[-1] == 3, f"aux_true_imgs: (n, 64, 64, 3), {aux_true_imgs.shape}"
            # advected frame t (frame t+1)
            advected_aux_gen = np.array(
                [src.advect(sample) for sample in aux_gen_imgs])  # 4D (n, h, w, m) (m: rho, vx, vy)
            assert advected_aux_gen.shape[-1] == 1, f"advected_aux_gen: (n, 64, 64, 1), {advected_aux_gen.shape}"
            advected_aux_truth = np.array([src.advect(sample) for sample in aux_true_imgs])  # 4D
            assert advected_aux_truth.shape[-1] == 1, f"advected_aux_truth: (n, 64, 64, 1), {advected_aux_truth.shape}"

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
            test_batch = gan_test[:n, :, :, 1:, 0]  # frame 1 to t (0 is not used bc. its only used in advection), 4D
            test_truth = gan_test_truth[:n, :, :, :, 0]  # 4th dim is always 1 so ":" is OK
        else:
            test_batch = gan_test[:n]
            test_truth = gan_test_truth[:n]
        gen_imgs = self.generator.predict(test_batch)
        fig, axs = plt.subplots(n, 3, figsize=(16, 16))
        for i in range(n):
            vmax = np.max([np.max(test_batch[i]), np.max(test_truth[i])])
            im = axs[i, 0].imshow(test_batch[i, :, :, 0], vmax=vmax)
            axs[i, 0].axis('off')
            src.colorbar(im)
            axs[i, 0].set_title("Frame t")
            im2 = axs[i, 1].imshow(test_truth[i, :, :, 0], vmax=vmax)
            axs[i, 1].axis('off')
            src.colorbar(im2)
            axs[i, 1].set_title("Frame t+1")
            im3 = axs[i, 2].imshow(gen_imgs[i, :, :, 0], vmax=vmax)
            axs[i, 2].axis('off')
            src.colorbar(im3)
            axs[i, 2].set_title("Prediction t+1")
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

    def update_loss_weights(self):
        self.loss_weights[0] -= 1
        self.loss_weights[1:] = [x + 1 for x in self.loss_weights[1:]]
        print(f"Updated loss weights: {self.loss_weights}")
        self.combined.compile(loss=self.losses, optimizer=self.g_optimizer, loss_weights=self.loss_weights)
