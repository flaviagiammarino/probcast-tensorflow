from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, GRU, Dense, Concatenate

def generator(gru_units, dense_units, sequence_length, noise_dimension, model_dimension):

    '''
    Generator model, see Figure 1 in the ProbCast paper.

    Parameters:
    __________________________________
    gru_units: list.
        Number of hidden units of each GRU layer.

    dense_units: int.
        Number of hidden units of the dense layer.

    sequence_length: int.
        Number of past time steps used as input.

    noise_dimension: int.
        Dimension of the noise vector concatenated to the outputs of the GRU block.

    model_dimension: int.
        Number of time series.
    '''

    # Input sequence.
    inputs = Input(shape=(sequence_length, model_dimension))

    # GRU block.
    outputs = GRU(units=gru_units[0], return_sequences=False if len(gru_units) == 1 else True)(inputs)
    for i in range(1, len(gru_units)):
        outputs = GRU(units=gru_units[i], return_sequences=True if i < len(gru_units) - 1 else False)(outputs)

    # Noise vector.
    noise = Input(shape=noise_dimension)
    outputs = Concatenate(axis=-1)([noise, outputs])

    # Dense layers.
    outputs = Dense(units=dense_units)(outputs)
    outputs = Dense(units=model_dimension)(outputs)

    return Model([inputs, noise], outputs)


def discriminator(gru_units, dense_units, sequence_length, model_dimension):

    '''
    Discriminator model, see Figure 2 in the ProbCast paper.

    Parameters:
    __________________________________
    gru_units: list.
        Number of hidden units of each GRU layer.

    dense_units: int.
        Number of hidden units of the dense layer.

    sequence_length: int.
        Number of past time steps used as input.

    model_dimension: int.
        Number of time series.
    '''

    # Input sequence.
    inputs = Input(shape=(sequence_length + 1, model_dimension))

    # GRU block.
    outputs = GRU(units=gru_units[0], return_sequences=False if len(gru_units) == 1 else True)(inputs)
    for i in range(1, len(gru_units)):
        outputs = GRU(units=gru_units[i], return_sequences=True if i < len(gru_units) - 1 else False)(outputs)

    # Dense layers.
    outputs = Dense(units=dense_units)(outputs)
    outputs = Dense(units=1)(outputs)

    return Model(inputs, outputs)
