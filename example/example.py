import numpy as np
from probcast_tensorflow.model import ProbCast

# Generate two time series
N = 200
t = np.linspace(0, 1, N)
e = np.random.multivariate_normal(mean=np.zeros(2), cov=np.eye(2), size=N)
a = 10 * np.cos(2 * np.pi * (10 * t - 0.5)) + 0.1 * e[:, 0]
b = 20 * np.cos(2 * np.pi * (20 * t - 0.5)) + 0.2 * e[:, 1]
y = np.hstack([a.reshape(- 1, 1), b.reshape(- 1, 1)])

# Fit the model
model = ProbCast(
    y=y,
    forecast_length=20,
    sequence_length=40,
    quantiles=[0.05, 0.25, 0.5, 0.75, 0.95],
    generator_gru_units=[32, 16],
    discriminator_gru_units=[32, 16],
    generator_dense_units=8,
    discriminator_dense_units=8,
    noise_dimension=100,
    noise_dispersion=10,
)

model.fit(
    learning_rate=0.001,
    batch_size=32,
    epochs=100,
    verbose=True
)

# Plot the in-sample predictions
predictions = model.predict(index=180)
fig = model.plot_predictions()
fig.write_image('predictions.png', width=750, height=650)

# Plot the out-of-sample forecasts
forecasts = model.forecast()
fig = model.plot_forecasts()
fig.write_image('forecasts.png', width=750, height=650)