import numpy as np


class AvgPerceptronMulti():
    def __init__(self, eta=1, n_iter=3):
        # perceptron's eta = 1
        self.eta = eta
        self.n_iter = n_iter

    def fit(self, x, y, weight):
        # print("Initial weight: ", weight)
        weight_label = []
        # if it is training then initial weight == 0, but test case we have weight vector already
        if weight == 0:
            # feature per class(can get num of class by value, but use 10 (0,1,2,...,9))
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
            # print("Iteration ", iter_num)

            avg_f = []
            avg_f.append([weight_label, 1])

            # learning data
            for xt, yt in zip(x, y):
                score = []
                cls = 0
                for _ in range(10):
                    score.append(np.dot(xt, weight_label[cls]))
                    # print("class= ", cls, ", score= ", np.dot(xt, weight_label[cls]))
                    cls += 1

                prediction = np.array(score).argmax()

                if yt != prediction:
                    # correct label
                    weight_label[yt] += self.eta * xt
                    # wrong label
                    weight_label[prediction] -= self.eta * xt
                    # add new (w,c)
                    avg_f.append([weight_label, 1])
                    mis += 1
                else:
                    # (w,c+1)
                    avg_f[-1][1] += 1
                    no_mis += 1

            accuracy = no_mis / (mis + no_mis)

            result[0].append(iter_num)
            result[1].append(mis)
            result[2].append(accuracy)


            # average apply
            max_c = max(t[1] for t in avg_f)

            sum_w = np.zeros(np.array(weight_label).shape)
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
