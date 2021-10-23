import numpy as np


class AvgPerceptronBinary():
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

        # result will include num of mistake, accuracy, final weight
        result = []
        for _ in range(4):
            result.append([])

        # iteration
        for _ in range(self.n_iter):
            iter_num += 1
            mis = 0
            no_mis = 0

            avg_f = []
            avg_f.append([self.w, 1])

            # learning data
            for xt, yt in zip(x, y):
                prediction = np.sign(np.dot(xt, self.w))

                if yt != prediction:
                    self.w += self.eta * yt * xt
                    # add new (w,c)
                    avg_f.append([self.w, 1])
                    mis += 1
                else:
                    # (w,c+1)
                    avg_f[-1][1] += 1
                    no_mis += 1

            accuracy = no_mis / (mis + no_mis)

            result[0].append(iter_num)
            result[1].append(mis)
            result[2].append(accuracy)

            # print("Iteration ", iter_num)

            # average apply
            max_c = max(t[1] for t in avg_f)

            sum_w = np.zeros(self.w.shape)
            c = 0

            for i in avg_f:
                # print(i)
                if i[1] == max_c:
                    # print(i)
                    sum_w += i[0]
                    c += 1

            avg_weight = []

            # print("max count: ", max_c, ", number of max count: ", c)

            # print("Max count weight for iteration")
            # print(sum_w)

            # print("Last weight for iteration")
            # print(avg_f[-1][0])

            # print("max count weight and last weight are same")
            # print(sum_w == avg_f[-1][0])

            # if there are c number of max count
            for i in sum_w:
                avg_weight.append(i/c)

            # print("Final weight for iteration")
            # print(avg_weight)

            self.w = np.array(avg_weight)
            # print(self.w)

        result[3].append(self.w.tolist())

        return result
