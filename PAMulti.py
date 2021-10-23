import numpy as np


class PAMulti():
    def __init__(self, n_iter=3):
        self.n_iter = n_iter

    def fit(self, x, y, weight):
        # print("Initial weight: ", weight)
        weight_label = []
        # if it is training then initial weight == 0, but test case we have weight vector already
        if weight == 0:
            for _ in range(10):
                weight_label.append(np.zeros(x.shape[1]))
        else:
            weight_label = weight
        iter_num = 0

        # result will include num of mistake, accuracy, final weight
        result = []
        for _ in range(4):
            result.append([])

        # iteration
        for _ in range(self.n_iter):
            iter_num += 1
            mis = 0
            no_mis = 0

            # learning data
            for xt, yt in zip(x, y):
                score = []
                cls = 0

                # feature per class(can get num of class by value, but use 10 (0,1,2,...,9))
                for _ in range(10):
                    score.append(np.dot(xt, weight_label[cls]))
                    # print("class= ", cls, ", score= ", np.dot(xt, weight_label[cls]))
                    cls += 1

                prediction = np.array(score).argmax()

                # print("yt= ", yt, ", prediction= ", prediction)

                if yt != prediction:
                    self.eta = 1 - (weight_label[yt] - weight_label[prediction]) / np.sum(np.square(xt))
                    # correct label
                    weight_label[yt] += self.eta * xt
                    # wrong label
                    weight_label[prediction] -= self.eta * xt
                    mis += 1
                else:
                    no_mis += 1

            accuracy = no_mis / (mis + no_mis)

            result[0].append(iter_num)
            result[1].append(mis)
            result[2].append(accuracy)

            # print("Iteration ", iter_num)
            # print(self.w)

        result[3].append(weight_label)

        return result
