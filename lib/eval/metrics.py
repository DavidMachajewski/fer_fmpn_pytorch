import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument('--n_classes', default=7)
parser.add_argument('--ptf_cnfmat', help="path to .txt file with saved confusion matrix numpy format")
parser.add_argument('--save_to', help="path to save the results to")


class Metrics:
    """ Given a multiple class confusion matrix of the form
    TP FN
    FP TN
    where row i stands for the class i and the column j
    stands for the predicted class j
    this class computes the
    true positives
    false negatives
    false positives
    true negatives
    precision
    recall
    accuracy (per class)
    accuracy (overall model)
    """
    def __init__(self, args):
        self.n_classes = args.n_classes
        self.cnfmat = None

    def __init_results__(self):
        self.true_positives = []
        self.false_negatives = []
        self.false_positives = []
        self.true_negatives = []
        self.precision = []
        self.recall = []
        self.f1 = []
        self.accuracy = []
        self.accuracy_overall = None

    def __tp__(self, class_id):
        return self.cnfmat[class_id][class_id]

    def __fp__(self, class_id):
        tmptp = self.__tp__(class_id)
        return np.sum(self.cnfmat, axis=0)[class_id] - tmptp

    def __fn__(self, class_id):
        tmptp = self.__tp__(class_id)
        return np.sum(self.cnfmat, axis=1)[class_id] - tmptp

    def __tn__(self, class_id):
        return np.sum(self.cnfmat) - self.__tp__(class_id) - self.__fp__(class_id) - self.__fn__(class_id)

    def __precision__(self, class_id):
        tp = self.true_positives[class_id]
        fp = self.false_positives[class_id]
        prec = tp / (tp + fp)
        return prec

    def __recall__(self, class_id):
        tp = self.true_positives[class_id]
        fn = self.false_negatives[class_id]
        rec = tp / (tp + fn)
        return rec

    def __accuracy__(self, class_id):
        """This is the per class accuracy"""
        tp = self.true_positives[class_id]
        tn = self.true_negatives[class_id]
        fp = self.false_positives[class_id]
        fn = self.false_negatives[class_id]
        acc = (tp + tn) / (tp + fp + tn + fn)
        return acc

    def set_cnfmat(self, array):
        self.cnfmat = array

    def __f1__(self, class_id):
        f1 = 2 / (1 / self.recall[class_id] + 1 / self.recall[class_id])
        return f1

    def update(self):
        self.__init_results__()
        for i in range(self.n_classes):
            self.true_positives.append(self.__tp__(i))
            self.false_positives.append(self.__fp__(i))
            self.false_negatives.append(self.__fn__(i))
            self.true_negatives.append(self.__tn__(i))

            self.accuracy.append(self.__accuracy__(i))
            self.precision.append(self.__precision__(i))
            self.recall.append(self.__recall__(i))
            self.f1.append(self.__f1__(i))
        self.accuracy_overall = self.__accuracy_overall__()

    def __accuracy_overall__(self):
        """recall and precision become accuracy if we
        calculate theese metrics globally"""
        tmptp = np.sum(self.true_positives)
        print("tmps sum: ", tmptp)
        denominator = tmptp + np.sum(self.false_negatives)
        print("denom: ", denominator)
        return tmptp / denominator

    def from_npfile(self):
        # array has to match the sklearn conventions
        #
        # get confusion matrix from .txt file with np
        # use set_cnfmat
        cnfmat = np.loadtxt(args.ptf_cnfmat)
        self.from_array(cnfmat)

    def from_array(self, array):
        # array has to match the sklearn conventions of cnfmat :TODO: check again
        self.set_cnfmat(array)

    def results(self):
        self.update()
        print(self.cnfmat)
        return self.true_positives, self.false_positives, self.false_negatives, self.true_negatives, self.accuracy, self.precision, self.recall, self.accuracy_overall


if __name__ == '__main__':

    args = parser.parse_args()

    args.ptf_cnfmat = "F:/trainings2/fmpn\pretrained/1/run_fmpn_2021-05-26_10-44-11/test_fmpn_2021-05-26_10-44-11\plots\cnfmat.txt"

    C = np.array(
        [[6, 2, 3, 0, 0, 0, 1],
         [0, 6, 0, 0, 0, 0, 0],
         [0, 0, 15, 0, 0, 0, 0],
         [0, 0, 0, 4, 0, 0, 2],
         [0, 0, 0, 0, 18, 0, 0],
         [0, 0, 0, 0, 0, 6, 0],
         [0, 0, 0, 0, 0, 0, 27]])

    metrics = Metrics(args)
    # etrics.from_array(C)
    metrics.from_npfile()

    res = metrics.results()
    print(res)

    #
    # :TODO: F1 and means of values etc. ?
    #
