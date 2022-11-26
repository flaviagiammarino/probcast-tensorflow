import tensorflow as tf

def generator_loss(targets, predictions, prob_predictions):
    '''
    Generator loss.
    '''
    
    L1 = tf.keras.losses.MeanAbsoluteError()
    L2 = tf.keras.losses.BinaryCrossentropy()
    
    return 0.99 * L1(y_true=targets, y_pred=predictions) + 0.01 * L2(y_true=tf.ones_like(prob_predictions), y_pred=prob_predictions)


def discriminator_loss(prob_targets, prob_predictions):
    '''
    Discriminator loss.
    '''
    
    L = tf.keras.losses.BinaryCrossentropy()
    
    return L(y_true=tf.ones_like(prob_targets), y_pred=prob_targets) + L(y_true=tf.zeros_like(prob_predictions), y_pred=prob_predictions)