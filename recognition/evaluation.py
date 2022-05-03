from calibration.align import align_with_reference_board
from recognition.losses import hungarian_matching, loss_loc, extract_predict_and_target_fields
import tensorflow as tf
from sklearn.metrics import average_precision_score


def evaluate_multi_metric(predictions, ground_truths):
    """
    Evaluates the given predictions by computing detection loss, distance loss and field color loss.
    :param predictions: the predictions
    :param ground_truths: the ground truths
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


def dart_score_evaluation(images, predictions, ground_truths):
    """
    Determines average precision score of correctly classified darts.
    Assumes both predictions and ground_truths to be ordered wrt time of image capture for better board calibration.
    Must start with image of empty board.
    For now, only works for images of one camera.
        Expects 3 predictions per image of length 8:
        - entries 1 and 2 for classification of object/no-object, 1 means object is present
        - entries 3 and 4 entry for relative x and y positions of object
        - entries 5-8 for classification of field color the object is located in
    Expects 3 ground truths per image of length 4:
        - entry 1 is the object/no-object label
        - entries 2 and 3 are relative x and y positions
        - entry 4 is field color class label
        - entry 5 is dart score or -1 if no-object
    :param images: images
    :param predictions: predicted darts
    :param ground_truths: ground truths with true dart score
    :return:
    """
    matching = hungarian_matching(predictions, ground_truths)
    # rearrange ground truths according to matching
    ground_truths = tf.gather_nd(ground_truths, matching[:, :, tf.newaxis], batch_dims=1)

    pred_detect = tf.nn.softmax(predictions[:, :, :2], axis=-1)[:, :, 1].numpy()

    ground_truths = ground_truths.numpy()

    total_detections = []

    M = None
    for i in range(len(images)):
        image = images[i]
        ground_truth = ground_truths[i]
        # if no darts are on the image, compute next perspective transformation
        if len(ground_truth) == 0:
            M = align_with_reference_board(image)

            # add false positive detections:
            for j in range(len(pred_detect[i])):
                total_detections.append([pred_detect[i, j], 1])

        else:
            if M is None:
                raise Exception("Must start with image of emtpy board!")

            # transform predicted positions
            pass

            # determine score depending on position and detected color
            detected_scores = None
            pass

            # add detections
            for j in range(len(pred_detect[i])):
                ground_truth_score = ground_truth[j, 4]
                correct_score = 0 if detected_scores[j] == ground_truth_score else 1
                total_detections.append([pred_detect[i, j], correct_score])

    # compute average precision score
    ap = None

    return ap