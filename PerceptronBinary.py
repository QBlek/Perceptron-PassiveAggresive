import numpy as np


class PerceptronBinary():
    def __init__(self, eta=1, n_iter=3):
        # perceptron's eta = 1
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y, weight):
        # print("Initial weight: ", weight)
        # if it is training then initial weight == 0, but test case we have weight vector already
        if weight == 0:
            self.w = np.zeros(x.shape[1])
        else:
            self.w = np.array(weight)

        iter_num = 0
        # result will include iteration, num of mistake, accuracy, final weight
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
                prediction = np.sign(np.dot(xt, self.w))

                if yt != prediction:
                    self.w += self.eta * yt * xt
                    mis += 1
                else:
                    no_mis += 1

            accuracy = no_mis / (mis + no_mis)

            result[0].append(iter_num)
            result[1].append(mis)
            result[2].append(accuracy)

            # print("Iteration ", iter_num)
            # print(self.w)

        result[3].append(self.w.tolist())

        return result
