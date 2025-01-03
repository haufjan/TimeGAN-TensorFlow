import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
import tqdm
from tqdm import tqdm



#Define post-hoc discriminator
class Discriminator(keras.Model):
    def __init__(self,
                 seq_len: int,
                 dim: int,
                 hidden_dim: int,
                 epochs: int,
                 batch_size: int):
        super().__init__(name='Discriminator')

        self.seq_len = seq_len
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        self.rnn = keras.layers.GRU(units=hidden_dim)
        self.rnn.build((None, seq_len, dim))
        self.model = keras.layers.Dense(units=1, activation=None)
        self.model.build((None, hidden_dim))
        self.activation = keras.layers.Activation('sigmoid')
        self.activation.build((None, 1))

        self.loss_fn = keras.losses.BinaryCrossentropy(from_logits=True)
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        d_last_states = self.rnn(x)
        y_hat_logit = self.model(tf.transpose(d_last_states, perm=[0, 1]))
        y_hat = self.activation(y_hat_logit)
        return y_hat_logit, y_hat

def discriminative_score_metrics(original_data: np.ndarray, generated_data: np.ndarray):
    """Use post-hoc RNN to classify original data and synthetic data

    Args:
        - ori_data: original data
        - generated_data: generated synthetic data

    Returns:
        - discriminative_score: np.abs(classification accuracy - 0.5)
    """
    no, seq_len, dim = original_data.shape

    hidden_dim = int(dim/2)
    iterations = 2000
    batch_size = 128

    model = Discriminator(seq_len=seq_len,
                          dim=dim,
                          hidden_dim=hidden_dim,
                          epochs=iterations,
                          batch_size=batch_size)

    #Split data into train and test fractions
    x_train, x_test, x_hat_train, x_hat_test = train_test_split(original_data, generated_data, test_size=0.2)

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, x_hat_train)).cache().shuffle(x_train.shape[0])

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, x_hat_test)).cache().shuffle(x_test.shape[0])

    #Define train step
    @tf.function
    def train_step(x, x_hat):
        with tf.GradientTape() as tape:
            y_logit_real, _ = model(x)
            y_logit_fake, _ = model(x_hat)

            d_loss_real = tf.math.reduce_mean(model.loss_fn(tf.ones_like(y_logit_real, dtype=tf.float32),
                                                            y_logit_real))
            d_loss_fake = tf.math.reduce_mean(model.loss_fn(tf.zeros_like(y_logit_fake, dtype=tf.float32),
                                                            y_logit_fake))
            d_loss = d_loss_real + d_loss_fake
        
        grad = tape.gradient(d_loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))

    #Start training
    for itt in tqdm(range(model.epochs)):
        #Mini-batch training
        for x, x_hat in ds_train.batch(model.batch_size).prefetch(tf.data.AUTOTUNE):
            train_step(x, x_hat)
    
    #Define test step
    @tf.function
    def test_step(x_test, x_hat_test):
        _, y_pred_real = model(x_test)
        _, y_pred_fake = model(x_hat_test)

        return y_pred_real, y_pred_fake

    for x, x_hat in ds_test.batch(x_test.shape[0]).prefetch(tf.data.AUTOTUNE):
        y_pred_real, y_pred_fake = test_step(x, x_hat)

        y_pred_final = np.squeeze(np.concatenate((y_pred_real.numpy(), y_pred_fake.numpy()), axis=0))
        y_label_final = np.concatenate((np.ones([len(y_pred_real,)]), np.zeros([len(y_pred_fake,)])), axis=0)

        acc = accuracy_score(y_label_final, (y_pred_final > 0.5))
        discriminative_score = np.abs(acc - 0.5)

    return discriminative_score