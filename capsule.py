import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data


# read & load mnist data
mnist = input_data.read_data_sets("./data/")




"""1. Start"""

x = tf.placeholder(shape=[None, 28, 28, 1], dtype=tf.float32, name="x")




"""2. Set Primary Caps"""

# (32 * 6 * 6) , 8D vector
PrimaryCap_maps = 32
PrimaryCap_caps = PrimaryCap_maps * 6 * 6
PrimaryCap_dims = 8

# Set two convolutional layers & parameters

# 28 * 28 -> 20 * 20 (kernel_size=9, strides=1)
conv1_params = {
    "filters" : 256,
    "kernel_size" : 9,
    "strides" : 1,
    "padding" : "valid",
    "activation" : tf.nn.relu,
}

# 20 * 20 -> [(20-9)/2]+1 = 6, 6 * 6 (kernel_size=9, strides=2)
conv2_params = {
    "filters" : PrimaryCap_maps * PrimaryCap_dims,
    "kernel_size" : 9,
    "strides" : 2,
    "padding" : "valid",
    "activation" : tf.nn.relu,
}

conv1 = tf.layers.conv2d(x, name="conv1", **conv1_params)
conv2 = tf.layers.conv2d(conv1, name="conv2", **conv2_params)


# (batch, 20, 20, 256) -> (batch, 6, 6, 256) -> (batch, 6, 6, 32, 8) -> (batch, 1152, 8)
PrimatyCap_input = tf.reshape(conv2, [-1, PrimaryCap_caps, PrimaryCap_dims], name="PrimatyCap_input")


# def squash(s) // def epsilon for making norm non-zero during training
def squash(s, axis=1, epsilon=1e-7, name=None):

    with tf.name_scope(name, default_name="squash"):

        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=True)
        norm = tf.sqrt(squared_norm + epsilon)
        squash_w = squared_norm / (1. + squared_norm)
        unit_vector = s / norm

        return squash_w * unit_vector

PrimaryCap_output = squash(PrimatyCap_input, name="PrimaryCap_output")
# Tensor("PrimaryCap_output/mul:0", shape=(?, 1152, 8), dtype=float32)




"""3. Digit Caps"""

DigitCap_caps = 10
DigitCap_dims = 16

stddev = 0.01

W_init = tf.random_normal(
    shape=(1, PrimaryCap_caps, DigitCap_caps, DigitCap_dims, PrimaryCap_dims), # (1, 1152, 10, 16, 8) // u_1~u_1152 -> 10 index
    stddev=stddev, dtype=tf.float32, name="W_init"
)

W = tf.Variable(W_init, name="W")

batch_size = tf.shape(x)[0]
W_tiled = tf.tile(W, [batch_size, 1, 1, 1, 1], name="W_tiled")
# Tensor("W_tiled:0", shape=(?, 1152, 10, 16, 8), dtype=float32)

# expand tensor at tail
PrimaryCap_output_expanded = tf.expand_dims(PrimaryCap_output, -1, name="PrimaryCap_output_expanded")
# Tensor("PrimaryCap_output_expanded:0", shape=(?, 1152, 8, 1), dtype=float32)

PrimaryCap_output_tile = tf.expand_dims(PrimaryCap_output_expanded, 2, name="PrimaryCap_output_tile")
# Tensor("PrimaryCap_output_tile:0", shape=(?, 1152, 1, 8, 1), dtype=float32)

# PrimaryCapsule output Tiling
PrimaryCap_output_tiled = tf.tile(PrimaryCap_output_tile, [1, 1, DigitCap_caps, 1, 1], name="PrimaryCap_output_tiled")
# Tensor("PrimaryCap_output_tiled:0", shape=(?, 1152, 10, 8, 1), dtype=float32)

# ( 1152 * 10 ), 16D capsules
DigitCap_predicted = tf.matmul(W_tiled, PrimaryCap_output_tiled, name="DigitCap_predicted")
# Tensor("DigitCap_predicted:0", shape=(?, 1152, 10, 16, 1), dtype=float32




"""4. Routing Algorithm""" # suggest routing algorithm 1~3 step

"""step1"""

# set b_i,j = zero, ( ?, 1152, 10, 1, 1)
routing_b = tf.zeros([batch_size, PrimaryCap_caps, DigitCap_caps, 1, 1], dtype=np.float32, name="routing_b")

# c_i = softmax(b_i)
routing_c = tf.nn.softmax(routing_b, dim=2, name="routing_c")

# dot product c.dot(u), Auto-Broadcasting
routing_cu = tf.multiply(routing_c, DigitCap_predicted, name="routing_cu")
input_s = tf.reduce_sum(routing_cu, axis=1, keep_dims=True, name="input_s")
# Tensor("input_s:0", shape=(?, 1, 10, 16, 1), dtype=float32)

# v = squash(s)
DigitCap_output_step1 = squash(input_s, axis=-2, name="DigitCap_output_step1")
# Tensor("DigitCap_output_step1/mul:0", shape=(?, 1, 10, 16, 1), dtype=float32)


"""step2"""

DigitCap_output_step1_tiled = tf.tile(DigitCap_output_step1, [1, PrimaryCap_caps, 1, 1, 1], name="Digit_output_step1_tiled")
# Tensor("Digit_output_step1_tiled:0", shape=(?, 1152, 10, 16, 1), dtype=float32)

# v.dot(u)
agreement = tf.matmul(DigitCap_predicted, DigitCap_output_step1_tiled, transpose_a=True, name="agreement")
# Tensor("agreement:0", shape=(?, 1152, 10, 1, 1), dtype=float32)

routing_b_step2 = tf.add(routing_b, agreement, name="routing_b_step2")

routing_c_step2 = tf.nn.softmax(routing_b_step2, dim=2, name="routing_c_step2")

routing_cu_step2 = tf.multiply(routing_c_step2, DigitCap_predicted, name="routing_cu_step2")
input_s_step2 = tf.reduce_sum(routing_cu_step2, axis=1, keep_dims=True, name="input_s_step2")

DigitCap_output_step2 = squash(input_s_step2, axis=-2, name="DigitCap_output_step2")



"""step3"""

DigitCap_output_step2_tiled = tf.tile(DigitCap_output_step2, [1, PrimaryCap_caps, 1, 1, 1], name="Digit_output_step2_tiled")

agreement_ = tf.matmul(DigitCap_predicted, DigitCap_output_step2_tiled, transpose_a=True, name="agreement_")

routing_b_step3 = tf.add(routing_b_step2, agreement_, name="routing_b_step3")

routing_c_step3 = tf.nn.softmax(routing_b_step3, dim=2, name="routing_c_step3")

routing_cu_step3 = tf.multiply(routing_c_step3, DigitCap_predicted, name="routing_cu_step3")
input_s_step3 = tf.reduce_sum(routing_cu_step3, axis=1, keep_dims=True, name="input_s_step3")

DigitCap_output_step3 = squash(input_s_step3, axis=-2, name="DigitCap_output_step3")



DigitCap_output = DigitCap_output_step3
# Tensor("DigitCap_output_step3/mul:0", shape=(?, 1, 10, 16, 1), dtype=float32)



"""5. Estimated class probabilities"""

def norm(s, axis=-1, epsilon=1e-7, keep_dims=False, name=None):
    with tf.name_scope(name, default_name="norm"):
        squared_norm = tf.reduce_sum(tf.square(s), axis=axis, keep_dims=keep_dims)
        return tf.sqrt(squared_norm + epsilon)

y_p = norm(DigitCap_output, axis=-2, name="y_p")
# Tensor("y_p/Sqrt:0", shape=(?, 1, 10, 1), dtype=float32)

# finding DigitCap_output index 0~9
y_p_argmax = tf.argmax(y_p, axis=2, name="y_p")
# Tensor("y_p_1:0", shape=(?, 1, 1), dtype=int64)

y_pred = tf.squeeze(y_p_argmax, axis=[1,2], name="y_pred")
# Tensor("y_pred:0", shape=(?,), dtype=int64)



"""6. Labeling"""
y = tf.placeholder(shape=[None], dtype=tf.int64, name="y")


"""7. Calculate Margin Loss"""
m_plus = 0.9
m_minus = 0.1
lambda_ = 0.5
T = tf.one_hot(y, depth=DigitCap_caps, name="T")
# Tensor("T:0", shape=(?, 10), dtype=float32)

DigitCap_output_norm = norm(DigitCap_output, axis=-2, keep_dims=True, name="DigitCap_output_norm")

plus_error_v = tf.square(tf.maximum(0., m_plus - DigitCap_output_norm), name="plus_error_v")
plus_error = tf.reshape(plus_error_v, shape=(-1,10), name="plus_error")

minus_error_v = tf.square(tf.maximum(0., DigitCap_output_norm - m_minus), name="minus_error_v")
minus_error = tf.reshape(minus_error_v, shape=(-1, 10), name="minus_error")

L = tf.add(T * plus_error, lambda_ * (1.0 - T) * minus_error, name="L")
# Tensor("L:0", shape=(?, 10), dtype=float32)

margin_loss = tf.reduce_mean(tf.reduce_sum(L, axis=1), name="margin_loss")
# Tensor("margin_loss:0", shape=(), dtype=float32)




"""8. Reconstruction"""

mask_with_labels = tf.placeholder_with_default(False, shape=(), name="mask_with_labels")

reconstruction_targets = tf.cond(mask_with_labels,
                                 lambda: y,    # if True
                                 lambda: y_pred,    # if False
                                 name="reconstruction_targets")

reconstruction_mask = tf.one_hot(reconstruction_targets, depth=DigitCap_caps, name="reconstruction_mask")
# Tensor("reconstruction_mask:0", shape=(?, 10), dtype=float32)

reconstruction_mask_reshaped = tf.reshape(reconstruction_mask, [-1, 1, DigitCap_caps, 1, 1], name="reconstruction_mask_reshaped")
# Tensor("reconstruction_mask_reshaped:0", shape=(?, 1, 10, 1, 1), dtype=float32)

DigitCap_output_masked = tf.multiply(DigitCap_output, reconstruction_mask_reshaped, name="DigitCab_output_masked")
# Tensor("DigitCab_output_masked:0", shape=(?, 1, 10, 16, 1), dtype=float32)

decoder_input = tf.reshape(DigitCap_output_masked, [-1, DigitCap_caps * DigitCap_dims], name="decoder_input")
# Tensor("decoder_input:0", shape=(?, 160), dtype=float32)


"""9. Decoder"""

# two FC(Fully-connected) ReLU
hidden1 = 512
hidden2 = 1024
n_output = 28 * 28

with tf.name_scope("decoder"):
    hidden1 = tf.layers.dense(decoder_input, hidden1, activation=tf.nn.relu, name="hidden1")
    hidden2 = tf.layers.dense(hidden1, hidden2, activation=tf.nn.relu, name="hidden2")
    decoder_output = tf.layers.dense(hidden2, n_output, activation=tf.nn.sigmoid, name="decoder_output")


"""10. Reconstruction Loss"""

x_flat = tf.reshape(x, [-1, n_output], name="x_flat")
squared_difference = tf.square(x_flat - decoder_output, name="squared_difference")
reconstruction_loss = tf.reduce_sum(squared_difference, name="reconstruction_loss")

"""11. Final Loss"""

#scale down reconstruction loss
scale_down = 0.0005
loss = tf.add(margin_loss, scale_down * reconstruction_loss, name="loss")


"""12. Accuracy"""

correct_prediction = tf.equal(y, y_pred, name="correct_prediction")
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32), name="accuracy")


"""13. Training Operations"""

optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")


"""14. Def Initializer and Saver"""

init = tf.global_variables_initializer()
saver = tf.train.Saver()





"""15. Training"""

# 55000 data mnist.train / 10000 data mnist.test / 5000 data mnist.validation
epochs = 10
batch_size = 50
restore_checkpoint = True


iterations_per_epoch = mnist.train.num_examples // batch_size
iterations_validation = mnist.validation.num_examples // batch_size
min_loss_val = np.infty # infinity
ckpt_path = "./saved/"

# If you already trined and saved ckpt file, comment out the following : start --- end

start-----------------------------------------------------------------------------

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        if restore_checkpoint and tf.train.checkpoint_exists(ckpt_path):
            saver.restore(sess, ckpt_path)
        else:
            init.run()


        for epoch in range(epochs):

            for iteration in range(1, iterations_per_epoch + 1):
                x_batch, y_batch = mnist.train.next_batch(batch_size)
                # array shape [784], let reshape [28 * 28]

                _, loss_train = sess.run(
                    [training_op, loss],
                    feed_dict={x: x_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch,
                               mask_with_labels: True}
                )

                print("\rIteration: {}/{} ({:.1f}% Loss: {:.4f})".format(
                    iteration, iterations_per_epoch,
                    iteration * 100 / iterations_per_epoch,
                    loss_train
                ),
                    end="" # delete \n
                )

            loss_vals = []
            acc_vals = []

            for iteration in range(1, iterations_per_epoch + 1):
                x_batch, y_batch = mnist.validation.next_batch(batch_size)
                loss_val, acc_val = sess.run(
                    [loss, accuracy],
                    feed_dict={x: x_batch.reshape([-1, 28, 28, 1]),
                               y: y_batch}
                )

                loss_vals.append(loss_val)
                acc_vals.append(acc_val)
                print("\rEvaluating the model: {}/{} ({:.1f})".format(
                    iteration, iterations_per_epoch,
                    iteration * 100 / iterations_per_epoch
                ),
                    end=" " * 10
                )

            loss_val = np.mean(loss_vals)
            acc_val = np.mean(acc_vals)

            print("\rEpoch: {} Val accuracy: {:.4f}% Loss: {:.6f}{}".format(
                epoch + 1, acc_val * 100, loss_val,
                "decreased" if loss_val < best_loss_val else ""
            ))

            if loss_val < best_loss_val:
                save_path = saver.save(sess, ckpt_path)
                best_loss_val = loss_val


"""16. Evaluation"""

iterations_test = mnist.test.num_examples // batch_size

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, ckpt_path)

        loss_tests = []
        acc_tests = []

        for iteration in range(1, iterations_test + 1):
            x_batch, y_batch = mnist.test.next_batch(batch_size)
            loss_test, acc_test = sess.run(
                [loss, accuracy],
                feed_dict={x: x_batch.reshape([-1, 28, 28, 1]),
                           y: y_batch}
            )

            loss_tests.append(loss_test)
            acc_tests.append(acc_test)

            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, iterations_test,
                iteration * 100 / iterations_test
            ),
            end=" " * 10
            )

            loss_test = np.mean(loss_tests)
            acc_test = np.mean(acc_tests)

            print("\rFinal test accuracy: {:.4f}% Loss: {:.6f}".format(
                acc_test * 100, loss_test
            ))

# Final test accuracy: 99.5400% Loss: 0.164313 routing 2 steps
# Final test accuracy: 99.4800% Loss: 0.139580 routing 3 steps
#-----------------------------------------------------------------------------end


"""17. Prediction"""


samples = 10
sample_images = mnist.test.images[:samples].reshape([-1, 28, 28, 1])


with tf.Session() as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, ckpt_path)

        DigitCap_output_value, decoder_output_value, y_pred_value = sess.run(
            [DigitCap_output, decoder_output, y_pred],
            feed_dict={x: sample_images,
                       y: np.array([], dtype=np.int64)}
        )

sample_images = sample_images.reshape(-1, 28, 28)
reconstructions = decoder_output_value.reshape([-1, 28, 28])

plt.figure(figsize=(samples * 1.5, 5))
# figure size

for index in range(samples):

    plt.subplot(2, samples, index + 1)
    plt.imshow(sample_images[index], cmap="gray")
    plt.title("Label:" + str(mnist.test.labels[index]))

    plt.axis("off")

    plt.subplot(2, samples, index + 11)
    plt.title("Predicted:" + str(y_pred_value[index]))
    plt.imshow(reconstructions[index], cmap="gray")

    plt.axis("off")

plt.show()


"""Appendix. Tweak"""

def tweak(output_vectors, start=-0.25, stop=0.25, steps=11):

    step = np.linspace(start, stop, steps)
    pose_parameters = np.arange(DigitCap_dims)

    # tweaking init_vector
    tweaks = np.zeros([DigitCap_dims, steps, 1, 1, 1, DigitCap_dims, 1])
    tweaks[pose_parameters, :, 0, 0, 0, pose_parameters, 0] = step
    output_vectors_expanded = output_vectors[np.newaxis, np.newaxis]
    # (batch_size = 10, 1, 10, 16, 1) -> (1, 1, 10, 1, 10, 16, 1)
    return tweaks + output_vectors_expanded


steps = 11

tweaked_vec = tweak(DigitCap_output_value, steps=steps)
# (16, 11, 10, 1, 10, 16, 1)

# ( num_poses * steps * batch_size, 1, 10, 16, 1)
tweaked_vec_reshaped = tweaked_vec.reshape(
    [-1, 1, DigitCap_caps, DigitCap_dims, 1])

tweak_labels = np.tile(mnist.test.labels[:samples], DigitCap_dims * steps)

with tf.Session() as sess:
    with tf.device("/gpu:0"):
        saver.restore(sess, ckpt_path)

        decoder_output_value = sess.run(
                decoder_output,
                feed_dict={DigitCap_output: tweaked_vec_reshaped,
                           mask_with_labels: True,
                           y: tweak_labels})

tweak_reconstructions = decoder_output_value.reshape([DigitCap_dims, steps, samples, 28, 28])


for loop in range(5):
    print("Tweak MNIST #{}".format(loop))
    plt.figure(figsize=(steps, samples))

    for row in range(samples):

        for col in range(steps):

            plt.subplot(samples, steps, row * steps + col + 1)
            plt.imshow(tweak_reconstructions[loop, col, row], cmap="gray")
            plt.axis("off")

    plt.show()