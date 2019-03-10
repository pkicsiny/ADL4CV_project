import src
from tensorflow import keras


def unet(input_shape=(64, 64, 1), dropout=0.0, batchnorm=False, kernel_size=4, feature_mult=1, relu_coeff=0.1):
    init = keras.layers.Input(shape=input_shape)

    ConvDown1 = keras.layers.Conv2D(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(init)  # 32
    if batchnorm:
        ConvDown1 = keras.layers.BatchNormalization()(ConvDown1)
    Lr1 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown1)
    if (dropout > 0) and (dropout <= 1):
        Lr1 = keras.layers.Dropout(dropout)(Lr1)

    ConvDown2 = keras.layers.Conv2D(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr1)  # 16
    if batchnorm:
        ConvDown2 = keras.layers.BatchNormalization()(ConvDown2)
    Lr2 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown2)
    if (dropout > 0) and (dropout <= 1):
        Lr2 = keras.layers.Dropout(dropout)(Lr2)

    ConvDown3 = keras.layers.Conv2D(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr2)  # 8
    if batchnorm:
        ConvDown3 = keras.layers.BatchNormalization()(ConvDown3)
    Lr3 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown3)
    if (dropout > 0) and (dropout <= 1):
        Lr3 = keras.layers.Dropout(dropout)(Lr3)

    ConvDown4 = keras.layers.Conv2D(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr3)  # 4
    if batchnorm:
        ConvDown4 = keras.layers.BatchNormalization()(ConvDown4)
    Lr4 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown4)
    if (dropout > 0) and (dropout <= 1):
        Lr4 = keras.layers.Dropout(dropout)(Lr4)

    ConvDown5 = keras.layers.Conv2D(filters=64*feature_mult, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                    padding="same")(Lr4)  # 2
    if batchnorm:
        ConvDown5 = keras.layers.BatchNormalization()(ConvDown5)
    Lr5 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvDown5)
    if (dropout > 0) and (dropout <= 1):
        Lr5 = keras.layers.Dropout(dropout)(Lr5)

    ConvUp4 = keras.layers.Conv2DTranspose(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(Lr5)
    if batchnorm:
        ConvUp4 = keras.layers.BatchNormalization()(ConvUp4)
    Lr6 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp4)
    if (dropout > 0) and (dropout <= 1):
        Lr6 = keras.layers.Dropout(dropout)(Lr6)
    merge1 = keras.layers.concatenate([ConvDown4, Lr6], axis=-1)

    ConvUp3 = keras.layers.Conv2DTranspose(filters=32*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge1)
    if batchnorm:
        ConvUp3 = keras.layers.BatchNormalization()(ConvUp3)
    Lr7 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp3)
    if (dropout > 0) and (dropout <= 1):
        Lr7 = keras.layers.Dropout(dropout)(Lr7)
    merge2 = keras.layers.concatenate([ConvDown3, Lr7], axis=-1)

    ConvUp2 = keras.layers.Conv2DTranspose(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge2)
    if batchnorm:
        ConvUp2 = keras.layers.BatchNormalization()(ConvUp2)
    Lr8 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp2)
    if (dropout > 0) and (dropout <= 1):
        Lr8 = keras.layers.Dropout(dropout)(Lr8)
    merge3 = keras.layers.concatenate([ConvDown2, Lr8], axis=-1)

    ConvUp1 = keras.layers.Conv2DTranspose(filters=16*feature_mult, kernel_size=(kernel_size, kernel_size),
                                           strides=(2, 2), padding="same")(merge3)
    if batchnorm:
        ConvUp1 = keras.layers.BatchNormalization()(ConvUp1)
    Lr9 = keras.layers.LeakyReLU(alpha=relu_coeff)(ConvUp1)
    if (dropout > 0) and (dropout <= 1):
        Lr9 = keras.layers.Dropout(dropout)(Lr9)
    merge4 = keras.layers.concatenate([ConvDown1, Lr9], axis=-1)

    ConvUp0 = keras.layers.Conv2DTranspose(filters=1, kernel_size=(kernel_size, kernel_size), strides=(2, 2),
                                           padding="same", activation='tanh')(merge4)

    return keras.models.Model(inputs=init, outputs=ConvUp0)


def spatial_discriminator(input_shape=(64, 64, 1), condition_shape=(64, 64, 1),
                          dropout=0, batchnorm=False, wgan=False):
    """
    from tempoGAN paper(Appendix A): "BN denotes batch normalization, which is not used in the
    last layer of G, the first layer of Dt and the first layer of Ds [Radford et al. 2016]."
    """
    # condition is the frame t (the original frame) or the sequence of past frames
    condition = keras.layers.Input(shape=condition_shape)
    # other is the generated prediction of frame t+1 or the ground truth frame t+1
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([condition, other])

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(combined_imgs)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(conv1)
    if (dropout > 0) and (dropout <= 1):
        relu1 = keras.layers.Dropout(dropout)(relu1)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu1)
    if batchnorm:
        conv2 = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(conv2)
    if (dropout > 0) and (dropout <= 1):
        relu2 = keras.layers.Dropout(dropout)(relu2)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(relu2)
    if batchnorm:
        conv3 = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(conv3)
    if (dropout > 0) and (dropout <= 1):
        relu3 = keras.layers.Dropout(dropout)(relu3)

    conv4 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(relu3)
    if batchnorm:
        conv4 = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(conv4)
    if (dropout > 0) and (dropout <= 1):
        relu4 = keras.layers.Dropout(dropout)(relu4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    if not wgan:
        fcl1 = keras.layers.Activation('sigmoid', name="s_disc_output")(fcl1)

    return keras.models.Model(inputs=[condition, other], outputs=fcl1)


def temporal_discriminator(input_shape=(64, 64, 1), advected_shape=(64, 64, 1),
                           dropout=0.3, batchnorm=False, wgan=False):
    # A(G(x_{t-1})) or A(y_{t-1}) (A(frame t)=frame t+1)
    advected = keras.layers.Input(shape=advected_shape)
    # other is the generated prediction of t (frame t+1) or the ground truth of t (frame t+1)
    other = keras.layers.Input(shape=input_shape)
    # Concatenate image and conditioning image by channels to produce input
    combined_imgs = keras.layers.Concatenate(axis=-1)([advected, other])

    conv1 = keras.layers.Conv2D(filters=16, kernel_size=4, strides=2, padding='same')(combined_imgs)
    relu1 = keras.layers.LeakyReLU(alpha=0.2)(conv1)
    if (dropout > 0) and (dropout <= 1):
        relu1 = keras.layers.Dropout(dropout)(relu1)

    conv2 = keras.layers.Conv2D(filters=32, kernel_size=4, strides=2, padding='same')(relu1)
    if batchnorm:
        conv2 = keras.layers.BatchNormalization()(conv2)
    relu2 = keras.layers.LeakyReLU(alpha=0.2)(conv2)
    if (dropout > 0) and (dropout <= 1):
        relu2 = keras.layers.Dropout(dropout)(relu2)

    conv3 = keras.layers.Conv2D(filters=64, kernel_size=4, strides=2, padding='same')(relu2)
    if batchnorm:
        conv3 = keras.layers.BatchNormalization()(conv3)
    relu3 = keras.layers.LeakyReLU(alpha=0.2)(conv3)
    if (dropout > 0) and (dropout <= 1):
        relu3 = keras.layers.Dropout(dropout)(relu3)

    conv4 = keras.layers.Conv2D(filters=128, kernel_size=4, strides=2, padding='same')(relu3)
    if batchnorm:
        conv4 = keras.layers.BatchNormalization()(conv4)
    relu4 = keras.layers.LeakyReLU(alpha=0.2)(conv4)
    if (dropout > 0) and (dropout <= 1):
        relu4 = keras.layers.Dropout(dropout)(relu4)

    # Out: 1-dim probability
    flatten = keras.layers.Flatten()(relu4)
    fcl1 = keras.layers.Dense(1)(flatten)
    if not wgan:
        fcl1 = keras.layers.Activation('sigmoid', name="t_disc_output")(fcl1)

    return keras.models.Model(inputs=[advected, other], outputs=fcl1)

