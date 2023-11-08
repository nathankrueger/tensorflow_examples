import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.animation as animation

# how much data to we want to crunch?
N = 30
num_samples_per_class = 300

# how fast do we attempt to converge?
learning_rate = 0.11

# animation settings
interval = 150
frames = 2 * N

# 2 distinct point clouds
negative_samples = np.random.multivariate_normal(
    mean=[0, 3],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)
positive_samples = np.random.multivariate_normal(
    mean=[3, 0],
    cov=[[1, 0.5], [0.5, 1]],
    size=num_samples_per_class
)

# Combine the point clouds into an input tensor (2 * num_samples_per_class, 2)
inputs = np.vstack((negative_samples, positive_samples)).astype(np.float32)

# Create the target tensor (2 * num_samples_per_class, 1)
targets = np.vstack((np.zeros((num_samples_per_class, 1), dtype="float32"),
                     np.ones((num_samples_per_class, 1), dtype="float32")))

print(f"Inputs.shape: {inputs.shape}")
print(f"Targets.shape: {targets.shape}")

input_dim = inputs.shape[1]
output_dim = 1 # are we in the postive or negative samples -- a binary, 1D answer

# Weights for the model
W = tf.Variable(initial_value=tf.random.uniform(shape=(input_dim, output_dim)))

# Bias for the model
b = tf.Variable(initial_value=tf.zeros(shape=(output_dim,)))

"""Predict the output"""
def model(inputs):
    return tf.matmul(inputs, W) + b

"""Calcualte the loss"""
def square_loss(targets, predictions):
    per_sample_losses = tf.square(targets - predictions)
    return tf.reduce_mean(per_sample_losses)

"""Tune the weights & biases by one step"""
def train_one_step(inputs, targets, weights, biases):
    with tf.GradientTape() as tape:
        predictions = model(inputs)
        loss = square_loss(targets, predictions)
    grad_loss_wrt_W, grad_loss_wrt_B = tape.gradient(loss, [weights, biases])
    weights.assign_sub(grad_loss_wrt_W * learning_rate)
    biases.assign_sub(grad_loss_wrt_B * learning_rate)
    return loss

"""Update the plot for each frame"""
def update_plot(i, scat, class_line, points_sp, loss_sp, loss_line):
    # Move one step closer
    loss = train_one_step(inputs, targets, W, b)
    predictions = model(inputs)
    print(f"step {i}, loss: {loss}")

    # Massage the data for graphing purposes
    binary_predictions = tf.where(predictions > 0.5, tf.ones_like(predictions), tf.zeros_like(predictions))
    binary_correct_predictions = tf.where(binary_predictions == targets, tf.ones_like(predictions), tf.zeros_like(predictions))
    colors = binary_correct_predictions

    # Update the data for the line, and the colors of the points
    points_sp.set_title(f"Classified data (learning_rate: {learning_rate}, step: {i+1})")
    scat.set_array(colors[:,0])
    x = np.linspace(-1, 4, 100)
    y = (- W[0] / W[1]) * x + ((0.5 - b) / W[1])
    class_line[0].set_ydata(y)

    # Update the loss graph
    loss_line[0].set_ydata(np.append(loss_line[0].get_ydata(), loss.numpy()))
    loss_line[0].set_xdata(np.append(loss_line[0].get_xdata(), i))

    loss_sp.relim()
    loss_sp.autoscale_view()

# Show some results
predictions = model(inputs)
binary_predictions = tf.where(predictions > 0.5, tf.ones_like(predictions), tf.zeros_like(predictions))
binary_correct_predictions = tf.where(binary_predictions == targets, tf.ones_like(predictions), tf.zeros_like(predictions))
colors = binary_correct_predictions

fig = plt.figure(figsize=(10,6))
GridSpec = gridspec.GridSpec(ncols=2, nrows=1, figure=fig)
subfig_points = fig.add_subfigure(GridSpec[0])
subfig_loss = fig.add_subfigure(GridSpec[1])

subplot_points = subfig_points.subplots(1, 1)
subplot_points.set_title('Classification data')

subplot_loss = subfig_loss.subplots(1, 1)
subplot_loss.set_title('Loss vs. step')
subplot_loss.set_xlim(0, frames)
loss_line = subplot_loss.plot([], [], "-r")

# Add the raw point data
cmap = LinearSegmentedColormap.from_list('dont_care', [(1,0,0),(0,0.7,0)], N=2)
scat = subplot_points.scatter(inputs[:,0], inputs[:,1], c=colors, cmap=cmap)

# Add the line representing the classification function
x = np.linspace(-2, 6, 100)
y = (- W[0] / W[1]) * x + ((0.5 - b) / W[1])
classification_line = subplot_points.plot(x, y, "-b")
subplot_points.set_xlim(-2,6)
subplot_points.set_ylim(-2,6)

ani = animation.FuncAnimation(
    fig,
    update_plot,
    frames=range(frames),
    interval=interval,
    fargs=(scat, classification_line, subplot_points, subplot_loss, loss_line),
    repeat=False
)
learning_rate_str = str(learning_rate).replace('.', 'p')

# Uncomment to safe a gif of the animation
#ani.save(f"lin_classifier_rate_{learning_rate_str}.gif")

# Uncomment to instead draw the graph
plt.show()