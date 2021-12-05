import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.utils import timeseries_dataset_from_array
from tensorflow.keras.optimizers import Adam
pd.options.mode.chained_assignment = None

from probcast_tensorflow.modules import generator, discriminator, generator_loss, discriminator_loss
from probcast_tensorflow.plots import plot

class ProbCast():

    def __init__(self,
                 y,
                 sequence_length,
                 forecast_length,
                 quantiles=[0.1, 0.5, 0.9],
                 generator_gru_units=[64, 32],
                 generator_dense_units=16,
                 discriminator_gru_units=[64, 32],
                 discriminator_dense_units=16,
                 noise_dimension=100,
                 noise_dispersion=1):

        '''
        Implementation of multivariate time series forecasting model introduced in Koochali, A., Dengel, A.,
        & Ahmed, S. (2021). If You Like It, GAN It — Probabilistic Multivariate Times Series Forecast with GAN.
        In Engineering Proceedings (Vol. 5, No. 1, p. 40). Multidisciplinary Digital Publishing Institute.
        https://doi.org/10.3390/engproc2021005040

        Parameters:
        __________________________________
        y: np.array
            Target time series, array with shape (samples, targets) where samples is the length of the
            time series and targets is the number of target time series.

        sequence_length: int
            Number of past time steps to use as input.

        forecast_length: int.
            Number of future time steps to forecast.

        quantiles: list.
            Quantiles of target time series to forecast.

        generator_gru_units: list.
            The length of the list is the number of GRU layers in the generator, the items in the list are
            the number of hidden units of each layer.

        generator_dense_units: int.
            Number of hidden units of the dense layer in the generator.

        discriminator_gru_units: list.
            The length of the list is the number of GRU layers in the discriminator, the items in the list
            are the number of hidden units of each layer.

        discriminator_dense_units: int.
            Number of hidden units of the dense layer in the discriminator.

        noise_dimension: int.
            Dimension of the noise vector concatenated to the outputs of the GRU block in the generator.

        noise_dispersion: float.
            Standard deviation of the noise vector concatenated to the outputs of the GRU block in the generator.
        '''

        if type(y) != np.ndarray:
            raise ValueError('The target time series must be provided as a numpy array.')

        elif np.isnan(y).sum() != 0:
            raise ValueError('The target time series cannot contain missing values.')

        if len(y.shape) > 2:
            raise ValueError('The targets array cannot have more than 2 dimensions.')

        elif len(y.shape) == 1:
            y = np.expand_dims(y, axis=1)

        if type(generator_gru_units) != list:
            raise ValueError('The generator GRU units must be provided as a list.')

        if type(discriminator_gru_units) != list:
            raise ValueError('The discriminator GRU units must be provided as a list.')

        if type(quantiles) != list:
            raise ValueError('The quantiles must be provided as a list.')

        elif len(quantiles) == 0:
            raise ValueError('No quantiles were provided.')

        # Extract the quantiles.
        quantiles = np.unique(np.array(quantiles))
        if 0.5 not in quantiles:
            quantiles = np.sort(np.append(0.5, quantiles))

        # Scale the targets.
        mu, sigma = np.mean(y, axis=0), np.std(y, axis=0)
        y = (y - mu) / sigma

        # Save the inputs.
        self.y = y
        self.mu = mu
        self.sigma = sigma
        self.sequence_length = sequence_length
        self.forecast_length = forecast_length
        self.generator_gru_units = generator_gru_units
        self.generator_dense_units = generator_dense_units
        self.discriminator_gru_units = discriminator_gru_units
        self.discriminator_dense_units = discriminator_dense_units
        self.noise_dimension = noise_dimension
        self.noise_dispersion = noise_dispersion
        self.quantiles = quantiles
        self.samples = y.shape[0]
        self.targets = y.shape[1]

    def fit(self,
            learning_rate=0.001,
            batch_size=32,
            epochs=100,
            verbose=True):

        '''
        Train the model.

        Parameters:
        __________________________________
        learning_rate: float.
            Learning rate.

        batch_size: int.
            Batch size.

        epochs: int.
            Number of epochs.

        verbose: bool.
            True if the training history should be printed in the console, False otherwise.
        '''

        # Build the models.
        generator_model = generator(
            gru_units=self.generator_gru_units,
            dense_units=self.generator_dense_units,
            sequence_length=self.sequence_length,
            noise_dimension=self.noise_dimension,
            model_dimension=self.targets
        )

        discriminator_model = discriminator(
            gru_units=self.discriminator_gru_units,
            dense_units=self.discriminator_dense_units,
            sequence_length=self.sequence_length,
            model_dimension=self.targets
        )

        # Instantiate the optimizers.
        generator_optimizer = Adam(learning_rate=learning_rate)
        discriminator_optimizer = Adam(learning_rate=learning_rate)

        # Define the training loop.
        @tf.function
        def train_step(data):
            with tf.GradientTape() as generator_tape, tf.GradientTape() as discriminator_tape:

                # Extract the input sequences and target values.
                inputs = tf.cast(data[:, :-1, :], dtype=tf.float32)
                targets = tf.cast(data[:, -1:, :], dtype=tf.float32)

                # Generate the noise vector.
                noise = tf.random.normal(
                    mean=0.0,
                    stddev=self.noise_dispersion,
                    shape=(data.shape[0], self.noise_dimension)
                )

                # Generate the model predictions.
                outputs = generator_model([inputs, noise])
                outputs = tf.reshape(outputs, shape=(data.shape[0], 1, self.targets))

                # Pass the actual sequence and the predicted sequence to the discriminator.
                actual = discriminator_model(tf.concat([inputs, targets], axis=1))
                predicted = discriminator_model(tf.concat([inputs, outputs], axis=1))

                # Calculate the loss.
                g = generator_loss(targets, outputs)
                d = discriminator_loss(actual, predicted)

                # Calculate the gradient.
                dg = generator_tape.gradient(g, generator_model.trainable_variables)
                dd = discriminator_tape.gradient(d, discriminator_model.trainable_variables)

                # Update the weights.
                generator_optimizer.apply_gradients(zip(dg, generator_model.trainable_variables))
                discriminator_optimizer.apply_gradients(zip(dd, discriminator_model.trainable_variables))

                return g, d

        # Generate the training batches.
        dataset = timeseries_dataset_from_array(
            data=self.y,
            targets=None,
            sequence_length=self.sequence_length + 1,
            batch_size=batch_size
        )

        # Train the model.
        for epoch in range(epochs):
            for data in dataset:
                g, d = train_step(data)
            if verbose:
                print('Epoch: {}  Generator Loss: {:.8f}  Discriminator Loss: {:.8f}'.format(1 + epoch, g, d))

        # Save the model.
        self.model = generator_model

    def predict(self, index, simulations=100):

        '''
        Generate the in-sample predictions.

        Parameters:
        __________________________________
        index: int.
            The start index of the sequence to predict.

        simulations: int.
            The number of samples to generate for estimating the quantiles.

        Returns:
        __________________________________
        predictions: pd.DataFrame
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        if index < self.sequence_length:
            raise ValueError('The index must be greater than {}.'.format(self.sequence_length))

        elif index > len(self.y) - self.forecast_length:
            raise ValueError('The index must be less than {}.'.format(self.samples - self.forecast_length))

        # Generate the predictions for the selected sequence.
        outputs = np.zeros(shape=(simulations, self.forecast_length, self.targets))

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_dispersion,
            size=(simulations, self.forecast_length, self.noise_dimension)
        )

        for i in range(index, index + self.forecast_length):
            inputs = self.y[i - self.sequence_length: i, :]
            inputs = inputs.reshape(1, self.sequence_length, self.targets)
            inputs = np.repeat(inputs, simulations, axis=0)
            outputs[:, i - index, :] = self.model([inputs, noise[:, i - index, :]]).numpy()

        # Organize the predictions in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.quantiles[j]) for i in range(self.targets) for j in range(len(self.quantiles))])

        predictions = pd.DataFrame(columns=columns)
        predictions['time_idx'] = np.arange(self.samples)

        for i in range(self.targets):
            predictions['target_' + str(i + 1)] = self.mu[i] + self.sigma[i] * self.y[:, i]

            for j in range(len(self.quantiles)):
                predictions['target_' + str(i + 1) + '_' + str(self.quantiles[j])].iloc[index: index + self.forecast_length] = \
                    self.mu[i] + self.sigma[i] * np.quantile(outputs[:, :, i], q=self.quantiles[j], axis=0)

        predictions = predictions.astype(float)

        # Save the data frame.
        self.predictions = predictions

        # Return the data frame.
        return predictions

    def forecast(self, simulations=100):

        '''
        Generate the out-of-sample forecasts.

        Parameters:
        __________________________________
        simulations: int.
            The number of samples to generate for estimating the quantiles.

        Returns:
        __________________________________
        forecasts: pd.DataFrame
            Data frame including the actual values of the time series and the predicted quantiles.
        '''

        # Generate the forecasts.
        outputs = np.zeros(shape=(simulations, self.forecast_length, self.targets))

        noise = np.random.normal(
            loc=0.0,
            scale=self.noise_dispersion,
            size=(simulations, self.forecast_length, self.noise_dimension)
        )

        inputs = self.y[- self.sequence_length:, :]
        inputs = inputs.reshape(1, self.sequence_length, self.targets)
        inputs = np.repeat(inputs, simulations, axis=0)

        for i in range(self.forecast_length):
            outputs[:, i, :] = self.model([inputs, noise[:, i, :]]).numpy()
            inputs = np.append(inputs[:, 1:, :], outputs[:, i, :].reshape(simulations, 1, self.targets), axis=1)

        # Organize the forecasts in a data frame.
        columns = ['time_idx']
        columns.extend(['target_' + str(i + 1) for i in range(self.targets)])
        columns.extend(['target_' + str(i + 1) + '_' + str(self.quantiles[j]) for i in range(self.targets) for j in range(len(self.quantiles))])

        forecasts = pd.DataFrame(columns=columns)
        forecasts['time_idx'] = np.arange(self.samples + self.forecast_length)

        for i in range(self.targets):
            forecasts['target_' + str(i + 1)].iloc[: - self.forecast_length] = self.mu[i] + self.sigma[i] * self.y[:, i]

            for j in range(len(self.quantiles)):
                forecasts['target_' + str(i + 1) + '_' + str(self.quantiles[j])].iloc[- self.forecast_length:] = \
                    self.mu[i] + self.sigma[i] * np.quantile(outputs[:, :, i], q=self.quantiles[j], axis=0)

        forecasts = forecasts.astype(float)

        # Save the data frame.
        self.forecasts = forecasts

        # Return the data frame.
        return forecasts

    def plot_predictions(self):

        '''
        Plot the in-sample predictions.

        Returns:
        __________________________________
        go.Figure
        '''

        return plot(self.predictions, self.quantiles, self.targets)

    def plot_forecasts(self):

        '''
        Plot the out-of-sample forecasts.

        Returns:
        __________________________________
        go.Figure
        '''

        return plot(self.forecasts, self.quantiles, self.targets)
