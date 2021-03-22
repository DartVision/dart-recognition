from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
import tensorflow as tf
import numpy as np


def hungarian_matching(predictions, ground_truths, cost_pred=1, cost_loc=1):
    """
    Computes the optimal matching between predictions and ground truths like in "End-to-End Object Detection
    with Transformers"
    Assumes len(predictions) == len(ground_truths)
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
        target_labels = gt[:, 0].numpy().astype(np.int)

        # from now on we work in numpy
        # Detection costs.
        # 1 - prob_gt. 1 is omitted.
        pred_matrix = -pred_detect[:, target_labels]
        l2_distances = cdist(p[:, 2:].numpy(), gt[:, 1:].numpy(), metric='euclidean')

        total_cost_matrix = cost_loc * l2_distances + cost_pred * pred_matrix
        total_cost_matrix[:, target_labels == 1] = 1
        _, col_indices = linear_sum_assignment(total_cost_matrix)
        all_indices.append(col_indices)

    return tf.convert_to_tensor(all_indices)


def loss_loc(prediction, ground_truth):
    """
    Localization loss. Squared L2 loss
    :return:
    """
    pass


def loss_detect(prediction, label):
    """
    Detection loss. Softmax cross entropy
    :return:
    """
    pass


def hungarian_loss(prediction, ground_truth, mu=1):
    matching = None

    return mu * loss_loc(prediction[:, 2:], ground_truth[:, 1:]) + loss_detect(prediction[:, :2], ground_truth[:, 0])


if __name__ == '__main__':
    predictions = [[[0.4, 1.2, 0.8, 0.5],
                    [0.8, 0.2, 0.3, 0.3]]]
    predictions = tf.convert_to_tensor(predictions)

    ground_truths = [[[0, 0.3, 0.36],
                      [1, 0, 0]]]
    ground_truths = tf.convert_to_tensor(ground_truths)
    print(hungarian_matching(predictions, ground_truths).numpy())