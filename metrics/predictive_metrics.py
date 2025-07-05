import numpy as np
import tensorflow as tf
import keras
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm

class Predictor(keras.Model):
    """
    Post-hoc RNN to predict the next time step of the original data
    """
    def __init__(self,
                 seq_len: int,
                 dim: int,
                 hidden_dim: int,
                 epochs: int,
                 batch_size: int):        
        super().__init__(name='Predictor')

        # Attributes
        self.seq_len = seq_len
        self.dim = dim
        self.hidden_dim = hidden_dim
        self.epochs = epochs
        self.batch_size = batch_size

        # Layers
        self.rnn = keras.layers.GRU(units=hidden_dim, return_sequences=True)
        self.rnn.build((None, seq_len-1, dim-1))
        self.model = keras.layers.Dense(units=1, activation='sigmoid')
        self.model.build((None, seq_len-1, hidden_dim))

        # Loss function
        self.loss_fn = keras.losses.MeanAbsoluteError()

        # Optimizer
        self.optimizer = keras.optimizers.Adam(learning_rate=1e-3)

    def call(self, x):
        p_outputs = self.rnn(x)
        return self.model(p_outputs)

def predictive_score_metrics(original_data: np.ndarray, generated_data: np.ndarray) -> float:
    """
    Report the performance of Post-hoc RNN one-step ahead prediction
    """
    no, seq_len, dim = np.asarray(original_data).shape

    hidden_dim = int(dim/2)
    iterations = 5000
    batch_size = 128

    model = Predictor(seq_len=seq_len,
                      dim=dim,
                      hidden_dim=hidden_dim,
                      epochs=iterations,
                      batch_size=batch_size)
    
    # Prepare training   
    x_train = generated_data[:,:-1,:(model.dim-1)]
    y_train = np.reshape(generated_data[:,1:,(model.dim-1)], (generated_data.shape[0],generated_data.shape[1]-1,1))

    ds_train = tf.data.Dataset.from_tensor_slices((x_train, y_train)).cache().shuffle(x_train.shape[0])

    # Define train step
    @tf.function
    def train_step(x, y):
        with tf.GradientTape() as tape:
            pred_train = model(x)
        
            loss = model.loss_fn(y, pred_train)

        grad = tape.gradient(loss, model.trainable_variables)
        model.optimizer.apply_gradients(zip(grad, model.trainable_variables))
    
    # Start training
    for _ in tqdm(range(model.epochs)):
        # Mini-batch training on synthetic data
        for x, y in ds_train.batch(model.batch_size).prefetch(tf.data.AUTOTUNE):
            train_step(x, y)                    

    # Test the model on the original data      
    x_test = original_data[:,:-1,:(model.dim-1)]
    y_test = np.reshape(original_data[:,1:,(model.dim-1)], (original_data.shape[0],original_data.shape[1]-1,1))

    ds_test = tf.data.Dataset.from_tensor_slices((x_test, y_test)).cache().shuffle(x_test.shape[0])

    @tf.function
    def test_step(x):
        return model(x)

    # Compute predictive score as MAE
    mae = 0
    for x, y in ds_test.batch(x_test.shape[0]).prefetch(tf.data.AUTOTUNE):
        pred_test =  test_step(x)

        for i in range(len(pred_test)):
            mae += mean_absolute_error(y[i,:,:].numpy(), pred_test[i,:,:].numpy())

    return mae/no