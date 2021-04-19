from recognition.losses import hungarian_matching, loss_loc, extract_predict_and_target_fields
import tensorflow as tf
from sklearn.metrics import average_precision_score


def evaluate_multi_metric(predictions, ground_truths):
    """

    :param predictions:
    :param ground_truths:
    :return:
    """
    matching = hungarian_matching(predictions, ground_truths)
    # rearrange ground truths according to matching
    ground_truths = tf.gather_nd(ground_truths, matching[:, :, tf.newaxis], batch_dims=1)

    pred_detect = tf.nn.softmax(predictions[:, :, :2], axis=-1)[:, :, 1].numpy()
    gt_detect = ground_truths[:, :, 0].numpy()

    # Detection loss
    loss_detect = average_precision_score(gt_detect, pred_detect)

    # Distance loss
    loss_distance = loss_loc(predictions, ground_truths)

    # Field loss
    pred_field, target_field = extract_predict_and_target_fields(predictions, ground_truths)
    target_field = tf.one_hot(target_field, depth=pred_field.shape[-1])
    pred_field, target_field = pred_field.numpy(), target_field.numpy()
    loss_field = average_precision_score(target_field, pred_field)

    return loss_detect, loss_distance, loss_field
