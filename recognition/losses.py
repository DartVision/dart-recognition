from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tensorflow as tf
import numpy as np


def hungarian_matching(predictions, ground_truths, cost_pred=1, cost_loc=1, cost_field=0.1):
    """
    Computes the optimal matching between predictions and ground truths like in "End-to-End Object Detection
    with Transformers"
    Assumes len(predictions) == len(ground_truths)
    :param cost_field:
    :param cost_pred:
    :param cost_loc:
    :param predictions:
    :param ground_truths:
    :return:
    """
    all_indices = []
    for i in range(predictions.shape[0]):
        p = predictions[i]
        gt = ground_truths[i]
        pred_detect = tf.nn.softmax(p[:, :2], axis=-1).numpy()
        # ground truth labels
        target_object_labels = gt[:, 0].numpy().astype(np.int)

        target_field_labels = gt[:, 3].numpy().astype(np.int)
        pred_field = tf.nn.softmax(p[:, 4:8], axis=-1).numpy()

        # from now on we work in numpy
        # Detection costs.
        # 1 - prob_gt. 1 is omitted.
        field_matrix = -pred_field[:, target_field_labels]
        pred_matrix = -pred_detect[:, target_object_labels]
        l2_distances = cdist(p[:, 2:4].numpy(), gt[:, 1:3].numpy(), metric='euclidean')

        total_cost_matrix = cost_loc * l2_distances + cost_pred * pred_matrix + cost_field * field_matrix
        total_cost_matrix[:, target_object_labels == 0] = 0
        _, col_indices = linear_sum_assignment(total_cost_matrix)
        all_indices.append(col_indices)

    return tf.convert_to_tensor(all_indices, dtype=tf.int32)


def loss_loc(predictions, ground_truths):
    """
    Localization loss. Squared L2 loss for all predictions where the gt has an object
    :param: predictions: (bs, 3, 8) tensor containing predictions
    :param: ground truths: (bs, 3, 8) tensor containing ground truth annotations
    :return: location loss
    """
    target_class, target_location = ground_truths[:, :, :1], ground_truths[:, :, 1:3]
    diff = predictions[:, :, 2:4] - target_location
    diff = diff ** 2

    # Adjust shape of target_class
    target_class = tf.tile(target_class, multiples=(1, 1, diff.shape[-1]))
    # don't sum predictions with target_class == no_object
    distances = tf.reduce_sum(target_class * diff, axis=-1)
    return tf.reduce_mean(distances)


def loss_detect(predictions, ground_truths):
    """
    Detection loss. Softmax cross entropy.
    :return:
    """
    target_class = tf.cast(ground_truths[:, :, 0], tf.int32)
    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=predictions[:, :, :2])
    return tf.reduce_mean(cross_entropy)


def loss_field(predictions, ground_truths):
    """
    Field classification loss. Softmax cross entropy for all predictions where the gt has an object
    :param predictions:
    :param ground_truths:
    :param matching:
    :return:
    """
    predicted_class, target_class = extract_predict_and_target_fields(predictions, ground_truths)

    cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=target_class, logits=predicted_class)
    return tf.reduce_mean(cross_entropy)


def extract_predict_and_target_fields(predictions, ground_truths):
    # Calculate indices where gt has object
    indices = tf.where(tf.equal(ground_truths[:, :, 0], 1))

    target_class = tf.cast(ground_truths[:, :, 3], tf.int32)
    target_class = tf.gather_nd(target_class, indices)

    predicted_class = predictions[:, :, 4:8]
    predicted_class = tf.gather_nd(predicted_class, indices)

    return predicted_class, target_class


def hungarian_loss(predictions, ground_truths, mu=1, rho=1, global_step=None):
    """
    Hungarian loss similar to that in "End-to-End Object Detection with Transformers" by Carion et al.
    Expects 3 predictions per image of length 8:
        - entries 1 and 2 for classification of object/no-object, 1 means object is present
        - entries 3 and 4 entry for relative x and y positions of object
        - entries 5-8 for classification of field color the object is located in
    Expects 3 ground truths per image of length 4:
        - entry 1 is the object/no-object label
        - entries 2 and 3 are relative x and y positions
        - entry 4 is field color class label
    :param predictions: tensor of shape (bs, 3, 8)
    :param ground_truths: tensor of shape (bs, 3, 4)
    :param mu:
    :return:
    """
    # calculate bi-partite matching
    matching = hungarian_matching(predictions, ground_truths)
    # rearrange ground truths according to matching
    ground_truths = tf.gather_nd(ground_truths, matching[:, :, tf.newaxis], batch_dims=1)

    detection_loss = loss_detect(predictions, ground_truths)
    location_loss = loss_loc(predictions, ground_truths)
    field_loss = loss_field(predictions, ground_truths)

    tf.summary.scalar('train/detection', tf.reduce_mean(detection_loss), step=global_step.numpy())
    tf.summary.scalar('train/location', tf.reduce_mean(location_loss), step=global_step.numpy())
    tf.summary.scalar('train/field', tf.reduce_mean(field_loss), step=global_step.numpy())

    total_loss = detection_loss + mu * location_loss + rho * field_loss

    return total_loss


if __name__ == '__main__':
    predictions = [[
        [0.8, 0.2, 0.3, 0.3, 0, 0.9, 0.5, 0],
        [0.4, 1.2, 0.8, 0.5, 0, 0.5, 1, 0]
                    ]]
    predictions = tf.convert_to_tensor(predictions)

    ground_truths = [[[1.0, 1.0, 0.8, 2.],
                      [1., 0.3, 0.36, 1.]]]
    ground_truths = tf.convert_to_tensor(ground_truths)
    print(hungarian_loss(predictions, ground_truths).numpy())
