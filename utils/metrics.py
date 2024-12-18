# encoding: utf-8
import numpy as np
from options import args_parser
# from sklearn.metrics._ranking import roc_auc_score
from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score  # , sensitivity_score
from imblearn.metrics import sensitivity_score, specificity_score
import pdb
from sklearn.metrics._ranking import roc_auc_score

N_CLASSES = 10


# CLASS_NAMES = [ 'Melanoma', 'Melanocytic nevus', 'Basal cell carcinoma', 'Actinic keratosis', 'Benign keratosis']

def compute_metrics_test(gt, pred, n_classes=10):
    """
    Computes accuracy, precision, recall and F1-score from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
        competition: whether to use competition tasks. If False,
          use all tasks
    Returns:
        List of AUROCs of all classes.
    """

    # 기존 식
    gt_np = gt.cpu().detach().numpy()
    pred_np = pred.cpu().detach().numpy()

   

    #231128
    #gt, pred 숫자가 달라서 맞춰주는 식
    # 원하는 동일한 수로 맞출 값
    desired_value = 10
    # gt_np와 pred_np를 동일한 수로 맞춤
    scaling_factor = desired_value / gt_np.max()
    gt_np_scaled = gt_np * scaling_factor
    pred_np_scaled = pred_np * scaling_factor


    # 231128 cifar100 실행할때만 수정
    #gt_np1 = (gt_np)/10
    #pred_np1 = (pred_np)*10

    #231128
    indexes = range(n_classes)

    #231128
    #오류1
    #ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')

    #오류 2
    # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
    #AUROCs = roc_auc_score(gt_np, pred_np[:,1], multi_class='ovr')

    #오류 3
    # numpy.AxisError: axis 1 is out of bounds for array of dimension 1
    #AUROCs = roc_auc_score(gt_np, pred_np[:,-1], multi_class='ovr')
    
    #오류 4
    #ValueError: 'y_true' contains labels not in parameter 'labels'
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr', labels=np.arange(n_classes))

    #오류 5
    # ValueError: Number of given labels, 100, not equal to the number of columns in 'y_score', 10
    #try:
    #    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr', labels=np.unique(gt_np))
    #except ValueError as e:
    #    raise ValueError("Error in AUROC computation:", str(e))

    #오류 6
    # 라벨의 고유한 수와 'y_score'의 열 수가 일치하는지 확인하고, 일치하지 않으면 ValueError를 발생시킵니다.
    # gt_np = 100, pred_np = 10임.. 
    # gt_np와 pred_np를 맞춰야됨
    #try:
    #    # ensure the number of labels matches the number of columns in y_score
    #    if len(np.unique(gt_np)) != pred_np.shape[1]:
    #        raise ValueError(f"Number of given labels ({len(np.unique(gt_np))}) "
    #                         f"not equal to the number of columns in 'y_score' ({pred_np.shape[1]})")

    #    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr', labels=np.unique(gt_np))
    #except ValueError as e:
    #    raise ValueError("Error in AUROC computation:", str(e))

    # 오류 7
    #ValueError: continuous format is not supported
    #AUROCs = roc_auc_score(gt_np1, pred_np, multi_class='ovr')
    
    # 오류 8
    #ValueError: Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
    #AUROCs = roc_auc_score(gt_np, pred_np1, multi_class='ovr')

    # 오류 9
    # ensure that the number of classes is set correctly
    # 수정된 부분: multi_class를 'ovr'대신에 실제 클래스 수로 설정
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr', n_classes=n_classes)

    # 오류 10
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovo')

    # 오류 11
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='multinomial')

    # 오류 12
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='raise')

    # ovr = one vs rest
    # ova = one vs all
    # ovo = one vs one

    # 오류 13
    # ValueError: Target scores need to be probabilities for multiclass roc_auc, i.e. they should sum up to 1.0 over classes
    #AUROCs = roc_auc_score(gt_np, (pred_np)*10, multi_class='ovr')

    # 오류 14
    #ValueError: Number of classes in y_true not equal to the number of columns in 'y_score'
    # 각 총행의 총 합이 1이 되도록 확률로 변환
    #normalized_pred = (pred_np * 10) / np.sum((pred_np * 10), axis=1, keepdims=True)
    # roc_auc_score 함수에 변환된 y_socre 전달
    #AUROCs = roc_auc_score(gt_np, normalized_pred, multi_class='ovr')

    # 오류 15
    # 원-핫 인코딩으로 변환
    # n_classes = 100으로 수정
    # IndexError: arrays used as indices must be of integer (or boolean) type
    #gt_one_hot = np.eye(n_classes)[gt_np]
    #AUROCs = roc_auc_score(gt_one_hot, pred_np, multi_class='ovr')

    # 오류 16
    #IndexError: arrays used as indices must be of integer (or boolean) type
    # gt_np가 원-핫 인코딩 되어 있다면. 정수 인덱스로 변환
    #if gt_np.shape[1] == n_classes:
    #    gt_np = np.argmax(gt_np, axis=1)
    #
    # 원-핫 인코딩으로 변환
    #gt_one_hot = np.eye(n_classes)[gt_np]
    #
    #AUROCs = roc_auc_score(gt_one_hot, pred_np, multi_class='ovr')

    # 오류 17
    #numpy.AxisError: axis 1 is out of bounds for array of dimension 1
    # gt_np가 원-핫 인코딩 형태인 경우 정수 인덱스로 변환
    #if len(gt_np.shape) > 1 and gt_np.shape[1] > 1:
    #    gt_np = np.argmax(gt_np, axis=1)
    #
    # pred_np가 확률 점수 형태인 경우 최대 확률의 인덱스로 변환
    #if len(pred_np.shape) > 1 and pred_np.shape[1] > 1:
    #    pred_np = np.argmax(pred_np, axis=1)
    #
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')

    ## 오류 18
    ## IndexError: arrays used as indices must be of integer (or boolean) type
    ## gt_np가 2차원 배열인 경우에만 np.argmax 적용
    #if gt_np.ndim == 2 and gt_np.shape[1] == n_classes:
    #    gt_np = np.argmax(gt_np, axis=1)

    ## pred_np가 확률 점수 형태인 경우 최대 확률의 인덱스로 변환
    #if pred_np.ndim == 2 and pred_np.shape[1] == n_classes:
    #    pred_np = np.argmax(pred_np, axis=1)

    ## AUROC 계산은 원-핫 인코딩 또는 확률 점수를 필요로 하므로
    ## 원-핫 인코딩으로 변환
    #gt_one_hot = np.zeros((gt_np.size, n_classes))
    #gt_one_hot[np.arange(gt_np.size), gt_np] = 1

    #AUROCs = roc_auc_score(gt_one_hot, pred_np, multi_class='ovr')

    #Accus = accuracy_score(gt_np, pred_np)
    #Pre = precision_score(gt_np, pred_np, average='macro')
    #Recall = recall_score(gt_np, pred_np, average='macro')


    ## 오류 19
    ##IndexError: index 49 is out of bounds for axis 1 with size 10
    ## gt_np가 정수형이 아닌 경우 정수형으로 변환
    #if not np.issubdtype(gt_np.dtype, np.integer):
    #    gt_np = gt_np.astype(int)

    ## pred_np가 확률 점수 형태인 경우 최대 확률의 인덱스로 변환
    #if pred_np.ndim == 2 and pred_np.shape[1] == n_classes:
    #    pred_np = np.argmax(pred_np, axis=1)

    ## 원-핫 인코딩 배열 생성
    #gt_one_hot = np.zeros((gt_np.size, n_classes))
    #rows = np.arange(gt_np.size)
    #gt_one_hot[rows, gt_np] = 1

    ## 성능 케트릭 계산
    #AUROCs = roc_auc_score(gt_one_hot, pred_np, multi_class='ovr')

    #Accus = accuracy_score(gt_np, pred_np)
    #Pre = precision_score(gt_np, pred_np, average='macro')
    #Recall = recall_score(gt_np, pred_np, average='macro')

    #231211
    # cifar100 train 코드
    #
    AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')
    Accus = accuracy_score(gt_np, np.argmax(pred_np, axis=1))
    Pre = precision_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    Recall = recall_score(gt_np, np.argmax(pred_np, axis=1), average='macro')
    #240125
    #F1 = f1_score(gt_np, np.argmax(pred_np, axis=1), average='macro')

    #231210까지 버전
    #
    #AUROCs = roc_auc_score(gt_np, pred_np, multi_class='ovr')
    #Accus = accuracy_score(gt_np, pred_np)
    #Pre = precision_score(gt_np, pred_np, average='macro')
    #Recall = recall_score(gt_np, pred_np, average='macro')



    return AUROCs, Accus, Pre, Recall  # , Senss, Specs, Pre, F1


def compute_pred_matrix(gt, pred, n_classes):
    matrix = np.zeros([n_classes, n_classes])
    for idx_gt in range(len(gt)):
        matrix[int(gt[idx_gt])][pred[idx_gt]] += 1
    return matrix
