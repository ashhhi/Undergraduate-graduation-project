import tensorflow as tf

def dice_coefficient(y_true, y_pred, smooth=1e-7):
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    dice = (2.0 * intersection + smooth) / (union + smooth)
    return dice

def cross_entropy_loss(y_true, y_pred):
    cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
    return cross_entropy(y_true, y_pred)

def mixed_loss(y_true, y_pred):
    dice = dice_coefficient(y_true, y_pred)
    cross_entropy = cross_entropy_loss(y_true, y_pred)
    loss = 0.5 * dice + 0.5 * cross_entropy
    return loss

def lovasz_softmax_loss(y_true, y_pred):
    # Flatten the predictions and labels
    y_true_flat = tf.reshape(y_true, [-1])
    y_pred_flat = tf.reshape(y_pred, [-1])

    # Sort the predicted probabilities in descending order
    y_pred_sorted, idxs = tf.nn.top_k(y_pred_flat, k=tf.shape(y_pred_flat)[0])

    # Compute the Hinge loss
    errors = 1.0 - tf.multiply(y_true_flat, y_pred_sorted)
    errors_sorted = tf.sort(errors, direction='DESCENDING')

    # Compute the Lovasz-Softmax loss
    lovasz_loss = tf.reduce_mean(errors_sorted)

    return lovasz_loss