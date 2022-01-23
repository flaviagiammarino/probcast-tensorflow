import tensorflow as tf
from tensorflow.keras.losses import MeanAbsoluteError, BinaryCrossentropy

def generator_loss(actual, predicted):

    '''
    Generator loss.

    Parameters:
    __________________________________
    actual: tf.Tensor.
        Actual values.

    predicted: tf.Tensor.
        Predicted values.
    '''

    L = MeanAbsoluteError()

    return L(y_true=actual, y_pred=predicted)


def discriminator_loss(actual, predicted):

    '''
    Discriminator loss.

    Parameters:
    __________________________________
    actual: tf.Tensor.
        Actual sequences.

    predicted: tf.Tensor.
        Predicted sequences.
    '''

    L = BinaryCrossentropy(from_logits=True)

    return L(y_true=tf.ones_like(actual), y_pred=actual) + L(y_true=tf.zeros_like(predicted), y_pred=predicted)
