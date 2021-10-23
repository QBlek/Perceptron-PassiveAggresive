import numpy as np
import pandas as pd
import PerceptronBinary as pba
import PABinary as paba
import matplotlib.pyplot as plt

if __name__ == '__main__':

    training_set = pd.read_csv('./fashion-mnist_train.csv')

    nlabel = training_set['label']
    npixels = training_set.iloc[:, 1:]

    nlabel_val = nlabel.values
    x = npixels.values
    y = []

    for yt in nlabel_val:
        if yt % 2 == 0:
            y.append(1)
        else:
            y.append(-1)

    print("(Training) Perceptron Binary:")
    pb = pba.PerceptronBinary(eta=1, n_iter=20)
    result1 = pb.fit(x, y, 0)

    # Perceptron- The number of mistake per iteration
    # print(result1[1])

    # Perceptron- The number of accuracy per iteration
    # print(result1[2])

    # Perceptron- Final weight
    # print(result1[3])
    # w1 = sum(result1[3], [])


    print("(Training) PA Binary:")
    pab = paba.PABinary(n_iter=20)
    result2 = pab.fit(x, y, 0)

    # PA- The number of mistake per iteration
    # print(result2[1])

    # PA - The number of accuracy per iteration
    # print(result2[2])

    # PA - Final weight
    # print(result2[3])
    # w2 = sum(result2[3], [])



    test_set = pd.read_csv('fashion-mnist_test.csv')

    tlabel = test_set['label']
    tpixels = test_set.iloc[:, 1:]

    tlabel_val = tlabel.values
    x2 = tpixels.values
    y2 = []

    for yt in tlabel_val:
        if yt % 2 == 0:
            y2.append(1)
        else:
            y2.append(-1)

    print("(Test) Perceptron Binary:")
    pb = pba.PerceptronBinary(eta=1, n_iter=20)
    result3 = pb.fit(x2, y2, w1)

    # Perceptron- The number of mistake per iteration
    # print(result3[1])

    # Perceptron- The number of accuracy per iteration
    # print(result3[2])

    # Perceptron- Final weight
    # print(result3[3])

    print("(Test) PA Binary:")
    pab = paba.PABinary(n_iter=20)
    result4 = pab.fit(x2, y2, w2)

    # PA- The number of mistake per iteration
    # print(result4[1])

    # PA - The number of accuracy per iteration
    # print(result4[2])

    # PA - Final weight
    # print(result4[3])

    print("")
    print("Final result")
    print("Perceptron Binary's TRAINING accuracy:", )
    print(result1[2])
    print("")
    print("Final result")
    print("PA Binary's TRAINING accuracy:", )
    print(result2[2])
    print("Final result")
    print("Perceptron Binary's TEST accuracy:", )
    print(result3[2])
    print("Final result")
    print("PA Binary's TEST accuracy:", )
    print(result4[2])

    # Accuracy curves
    plt.plot(result1[0], result1[2], label='Training_Perceptron')
    plt.plot(result2[0], result2[2], label='Training_PA')
    plt.plot(result3[0], result3[2], label='Test_Perceptron')
    plt.plot(result4[0], result4[2], label='Test_PA')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
