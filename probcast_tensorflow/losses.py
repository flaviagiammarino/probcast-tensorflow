import tensorflow as tf

def generator_loss(targets, predictions, prob_predictions):
    '''
    Generator loss.
    '''
    
    loss = tf.keras.losses.mean_squared_error(y_true=targets, y_pred=predictions)
    loss += tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(prob_predictions), y_pred=prob_predictions)
    
    return tf.reduce_mean(loss)


def discriminator_loss(prob_targets, prob_predictions):
    '''
    Discriminator loss.
    '''
    
    loss = tf.keras.losses.binary_crossentropy(y_true=tf.ones_like(prob_targets), y_pred=prob_targets)
    loss += tf.keras.losses.binary_crossentropy(y_true=tf.zeros_like(prob_predictions), y_pred=prob_predictions)
    
    return tf.reduce_mean(loss)