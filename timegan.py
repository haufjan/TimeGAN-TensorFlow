import time
import numpy as np
import tensorflow as tf
import keras

# Define TimeGAN's recurrent networks
class Embedder(keras.Sequential):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers):
        super().__init__(name='Embedder')
        assert module_name in ['gru', 'lstm']
        # Attributes
        self.module_name = module_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.add(keras.Input(shape=(None, input_dim)))
        for _ in range(num_layers):
            if module_name == 'gru':
                self.add(keras.layers.GRU(units=hidden_dim, return_sequences=True))
            elif module_name == 'lstm':
                self.add(keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.add(keras.layers.Dense(units=hidden_dim, activation='sigmoid'))

    def build(self, sequence_length: int):
        self.layers[0].input_shape = (sequence_length, self.input_dim)
        super().build()

    def call(self, x, training: bool = True):
        return super().call(x, training=training)

class Recovery(keras.Sequential):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers):
        super().__init__(name='Recovery')
        assert module_name in ['gru', 'lstm']
        # Attributes
        self.module_name = module_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.add(keras.Input(shape=(None, hidden_dim)))
        for _ in range(num_layers):
            if module_name == 'gru':
                self.add(keras.layers.GRU(units=hidden_dim, return_sequences=True))
            elif module_name == 'lstm':
                self.add(keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.add(keras.layers.Dense(units=input_dim, activation='sigmoid'))

    def build(self, sequence_length: int):
        self.layers[0].input_shape = (sequence_length, self.hidden_dim)
        super().build()

    def call(self, x, training: bool = True):
        return super().call(x, training=training)

class Generator(keras.Sequential):
    def __init__(self, module_name, input_dim, hidden_dim, num_layers):
        super().__init__(name='Generator')
        assert module_name in ['gru', 'lstm']
        # Attributes
        self.module_name = module_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.add(keras.Input(shape=(None, input_dim)))
        for _ in range(num_layers):
            if module_name == 'gru':
                self.add(keras.layers.GRU(units=hidden_dim, return_sequences=True))
            elif module_name == 'lstm':
                self.add(keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.add(keras.layers.Dense(units=hidden_dim, activation='sigmoid'))

    def build(self, sequence_length: int):
        self.layers[0].input_shape = (sequence_length, self.input_dim)
        super().build()

    def call(self, x, training: bool = True):
        return super().call(x, training=training)

class Supervisor(keras.Sequential):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__(name='Supervisor')
        assert module_name in ['gru', 'lstm']
        # Attributes
        self.module_name = module_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.add(keras.Input(shape=(None, hidden_dim)))
        for _ in range(num_layers):
            if module_name == 'gru':
                self.add(keras.layers.GRU(units=hidden_dim, return_sequences=True))
            elif module_name == 'lstm':
                self.add(keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.add(keras.layers.Dense(units=hidden_dim, activation='sigmoid'))

    def build(self, sequence_length: int):
        self.layers[0].input_shape = (sequence_length, self.hidden_dim)
        super().build()

    def call(self, x, training: bool = True):
        return super().call(x, training=training)

class Discriminator(keras.Sequential):
    def __init__(self, module_name, hidden_dim, num_layers):
        super().__init__(name='Discriminator')
        assert module_name in ['gru', 'lstm']
        # Attributes
        self.module_name = module_name
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers

        # Layers
        self.add(keras.Input(shape=(None, hidden_dim)))
        for _ in range(num_layers):
            if module_name == 'gru':
                # Bidirectional discriminator
                self.add(keras.layers.Bidirectional(keras.layers.GRU(units=hidden_dim, return_sequences=True)))
                # Unidirectional discriminator
                # self.add(keras.layers.GRU(units=hidden_dim, return_sequences=True))
            elif module_name == 'lstm':
                # Bidirectional discriminator
                self.add(keras.layers.Bidirectional(keras.layers.LSTM(units=hidden_dim, return_sequences=True)))
                # Unidirectional discriminator
                # self.add(keras.layers.LSTM(units=hidden_dim, return_sequences=True))
        self.add(keras.layers.Dense(units=1, activation=None))

    def build(self, sequence_length: int):
        self.layers[0].input_shape = (sequence_length, self.hidden_dim)
        super().build()

    def call(self, x, training: bool = True):
        return super().call(x, training=training)

# Define loss functions
@tf.function
def embedding_loss(x, x_tilde):
    """
    Compute reconstruction loss between original and recovered sequences.
    """
    return 10*tf.math.sqrt(keras.losses.MeanSquaredError()(x, x_tilde))

@tf.function
def supervised_loss(h, h_hat_supervise):
    """
    Compute supervised loss by comparing one-step ahead original latent vectors with supervised original vectors.
    """
    return keras.losses.MeanSquaredError()(h[:,1:,:], h_hat_supervise[:,:-1,:])

@tf.function
def generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat, gamma: int = 1):
    """
    Compute combined generator loss from multiple loss measures.
    """
    fake = tf.ones_like(y_fake, dtype=tf.float32)

    # 1. Unsupervised generator loss
    g_loss_u = keras.losses.BinaryCrossentropy(from_logits=True)(fake, y_fake)
    g_loss_u_e = keras.losses.BinaryCrossentropy(from_logits=True)(fake, y_fake_e)

    # 2. Supervised loss
    g_loss_s = keras.losses.MeanSquaredError()(h[:,1:,:], h_hat_supervise[:,:-1,:])

    # 3. Two moments
    g_loss_v1 = tf.math.reduce_mean(tf.math.abs(tf.math.sqrt(tf.math.reduce_std(x_hat, axis=0) + 1e-6) - tf.math.sqrt(tf.math.reduce_std(x, axis=0) + 1e-6)))
    g_loss_v2 = tf.math.reduce_mean(tf.math.abs(tf.math.reduce_mean(x_hat, axis=0) - tf.math.reduce_mean(x, axis=0)))
    g_loss_v = g_loss_v1 + g_loss_v2

    return g_loss_u + gamma*g_loss_u_e + 100*tf.math.sqrt(g_loss_s) + 100*g_loss_v

@tf.function
def discriminator_loss(y_real, y_fake, y_fake_e, gamma: int = 1):
    """
    Compute unsupervised discriminator loss.
    """
    fake = tf.zeros_like(y_fake, dtype=tf.float32)
    valid = tf.ones_like(y_real, dtype=tf.float32)

    # Unsupervised loss
    d_loss_real = keras.losses.BinaryCrossentropy(from_logits=True)(valid, y_real)
    d_loss_fake = keras.losses.BinaryCrossentropy(from_logits=True)(fake, y_fake)
    d_loss_fake_e = keras.losses.BinaryCrossentropy(from_logits=True)(fake, y_fake_e)

    return d_loss_real + d_loss_fake + d_loss_fake_e*gamma

# Define TimeGAN
class TimeGAN():
    def __init__(self, module_name: str = 'gru',
                 input_dim: int = 1,
                 hidden_dim: int = 24,
                 num_layers: int = 3,
                 epochs: int = 1000,
                 batch_size: int = 128,
                 learning_rate: float = 1e-3):

        # Attributes
        self.module_name = module_name
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.epochs = epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        # Networks
        self.embedder = Embedder(module_name, input_dim, hidden_dim, num_layers)
        self.recovery = Recovery(module_name, input_dim, hidden_dim, num_layers)
        self.generator = Generator(module_name, input_dim, hidden_dim, num_layers)
        self.supervisor = Supervisor(module_name, hidden_dim, num_layers-1)
        self.discriminator = Discriminator(module_name, hidden_dim, num_layers)

        # Loss functions
        self.embedding_loss = embedding_loss
        self.supervised_loss = supervised_loss
        self.generator_loss = generator_loss
        self.discriminator_loss = discriminator_loss

        # Optimizers
        self.optimizer_e_0 = keras.optimizers.Adam(learning_rate)
        self.optimizer_e = keras.optimizers.Adam(learning_rate)
        self.optimizer_s = keras.optimizers.Adam(learning_rate)
        self.optimizer_g = keras.optimizers.Adam(learning_rate)
        self.optimizer_d = keras.optimizers.Adam(learning_rate)

        # Auxiliary
        self.fitting_time = None
        self.losses = []

    def fit(self, data_training: np.ndarray):
        """
        TimeGAN training.
        """
        # Track training time
        self.fitting_time = time.time()

        # Build networks
        self.embedder.build(data_training.shape[1])
        self.recovery.build(data_training.shape[1])
        self.generator.build(data_training.shape[1])
        self.supervisor.build(data_training.shape[1])
        self.discriminator.build(data_training.shape[1])

        # Cast datatype
        data_training = np.float32(data_training)

        # Create TensorFlow data set from training data sequences
        ds_train = tf.data.Dataset.from_tensor_slices(data_training).cache().shuffle(data_training.shape[0])

        # Define 1st training phase, Embedder-Recovery training
        @tf.function
        def train_step_e0(x):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                x_tilde = self.recovery(h)

                loss_e = self.embedding_loss(x, x_tilde)

            grad_e = tape.gradient(loss_e, self.embedder.trainable_variables + \
                                   self.recovery.trainable_variables)
            self.optimizer_e_0.apply_gradients(zip(grad_e, self.embedder.trainable_variables + \
                                                   self.recovery.trainable_variables))

            return loss_e

        print('Start Embedder-Recovery Training')
        for epoch in range(self.epochs):
            loss_e_record = []
            # Mini-batch training
            for x in ds_train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE):
                loss_e = train_step_e0(x)
                loss_e_record.append(loss_e)

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('Epoch:', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs,
                      '| loss_e:', f'{np.mean(loss_e_record):12.9f}')

        print('Finished Embedder-Rcovery Training\n')

        # Define 2nd training phase, Supervised Loss Only
        @tf.function
        def train_step_s(x):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)

                loss_s = self.supervised_loss(h, h_hat_supervise)

            grad_s = tape.gradient(loss_s, self.generator.trainable_variables + \
                                   self.supervisor.trainable_variables)
            self.optimizer_s.apply_gradients(zip(grad_s, self.generator.trainable_variables + \
                                                 self.supervisor.trainable_variables))

            return loss_s

        # Reuse recent data set and shuffle
        ds_train = ds_train.shuffle(buffer_size=data_training.shape[0])

        print('Start Training on Supervised Loss only')
        for epoch in range(self.epochs):
            loss_s_record = []
            # Mini-batch training
            for x in ds_train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE):
                loss_s = train_step_s(x)
                loss_s_record.append(loss_s)

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('Epoch:', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs,
                      '| loss_s:', f'{np.mean(loss_s_record):12.9f}')

        print('Finished Training on Supervised Loss only\n')

        # Define 3rd training phase, Joint Training
        # Generator, Supervisor
        @tf.function
        def train_step_g(x, z):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                e_hat = self.generator(z)
                h_hat = self.supervisor(e_hat)
                h_hat_supervise = self.supervisor(h)
                x_hat = self.recovery(h_hat)
                y_fake = self.discriminator(h_hat)
                y_fake_e = self.discriminator(e_hat)

                loss_g = self.generator_loss(y_fake, y_fake_e, h, h_hat_supervise, x, x_hat)

            grad_g = tape.gradient(loss_g, self.generator.trainable_variables + \
                                   self.supervisor.trainable_variables)

            self.optimizer_g.apply_gradients(zip(grad_g, self.generator.trainable_variables + \
                                                 self.supervisor.trainable_variables))

            return loss_g

        # Embedder, Recovery
        @tf.function
        def train_step_e(x):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                h_hat_supervise = self.supervisor(h)
                x_tilde = self.recovery(h)

                loss_e = self.embedding_loss(x, x_tilde) + 0.1*self.supervised_loss(h, h_hat_supervise)

            grad_e = tape.gradient(loss_e, self.embedder.trainable_variables + \
                                   self.recovery.trainable_variables)

            self.optimizer_e.apply_gradients(zip(grad_e, self.embedder.trainable_variables + \
                                                 self.recovery.trainable_variables))

            return loss_e

        # Discriminator
        @tf.function
        def train_step_d(x, z):
            with tf.GradientTape() as tape:
                h = self.embedder(x)
                e_hat = self.generator(z)
                h_hat = self.supervisor(h)

                y_fake = self.discriminator(h_hat)
                y_real = self.discriminator(h)
                y_fake_e = self.discriminator(e_hat)

                loss_d = self.discriminator_loss(y_real, y_fake, y_fake_e)

            # Check loss thresold and optimize
            if loss_d > 0.15:
                grad_d = tape.gradient(loss_d,
                                       self.discriminator.trainable_variables)
                self.optimizer_d.apply_gradients(zip(grad_d,
                                                     self.discriminator.trainable_variables))

            return loss_d

        print('Start Joint Training')
        for epoch in range(self.epochs):
            loss_g_record = []
            loss_e_record = []
            # Optimize generating networks twice in one epoch
            for _ in range(2):
                # Extend data set by noise vectors sampled from uniform distribution
                ds_train = tf.data.Dataset.from_tensor_slices((data_training,
                                                               keras.random.uniform(data_training.shape))).cache().shuffle(data_training.shape[0])

                # Mini-batch training
                for x, z in ds_train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE):
                    loss_g = train_step_g(x, z)
                    loss_g_record.append(loss_g)

                    loss_e = train_step_e(x)
                    loss_e_record.append(loss_e)

            # Extend data set by noise vectors sampled from uniform distribution
            ds_train = tf.data.Dataset.from_tensor_slices((data_training,
                                                           keras.random.uniform(data_training.shape))).cache().shuffle(data_training.shape[0])

            # Optimize discriminating network once in one epoch
            loss_d_record = []
            # Mini-batch training
            for x, z in ds_train.batch(self.batch_size).prefetch(tf.data.AUTOTUNE):
                loss_d = train_step_d(x, z)
                loss_d_record.append(loss_d)

            self.losses.append([np.mean(loss_g_record), np.mean(loss_e_record), np.mean(loss_d_record)])

            if (epoch + 1) % (0.1*self.epochs) == 0:
                print('Epoch:', repr(epoch + 1).rjust(len(str(self.epochs))), 'of', self.epochs,
                      '| loss_g:', f'{np.mean(loss_g_record):12.9f}',
                      '| loss_e:', f'{np.mean(loss_e_record):12.9f}',
                      '| loss_d:', f'{np.mean(loss_d_record):12.9f}')

        print('Finished Joint Training')

        self.fitting_time = np.round(time.time() - self.fitting_time, 3)
        print('\nElapsed Training Time:', time.strftime('%Hh %Mmin %Ss', time.gmtime(self.fitting_time)), '\n')

    def transform(self, shape: tuple, training: bool = False):
        """TimeGAN sequences generation"""
        @tf.function
        def generate_step(z):
            e_hat = self.generator(z, training=training)
            h_hat = self.supervisor(e_hat, training=training)
            x_hat = self.recovery(h_hat, training=training)

            return x_hat

        # Data set holding noise vectors
        ds_noise = tf.data.Dataset.from_tensor_slices(keras.random.uniform(shape)).shuffle(shape[0])

        sequences_generated = []
        for z in ds_noise.batch(1).prefetch(tf.data.AUTOTUNE):
            x_hat = generate_step(z)

            sequences_generated.append(np.squeeze(x_hat.numpy()))

        return np.stack(sequences_generated)