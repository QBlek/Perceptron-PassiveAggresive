import pandas as pd
import PerceptronBinary as pba
import AvgPerceptronBinary as apba
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = pd.read_csv('./fashion-mnist_train.csv')

    label = dataset['label']
    pixels = dataset.iloc[:, 1:]

    label_val = label.values
    x = pixels.values
    y = []

    for yt in label_val:
        if yt % 2 == 0:
            y.append(1)
        else:
            y.append(-1)

    print("(Training) Plain Perceptron Binary:")
    pb = pba.PerceptronBinary(eta=1, n_iter=20)
    result1 = pb.fit(x, y, 0)

    # Perceptron- The number of mistake per iteration
    # print(result1[1])

    # Perceptron- The number of accuracy per iteration
    # print(result1[2])

    # Perceptron- Final weight
    # print(result1[3])
    w1 = sum(result1[3], [])

    print("(Training) Average Perceptron Binary:")
    apb = apba.AvgPerceptronBinary(eta=1, n_iter=20)
    result2 = apb.fit(x, y, 0)

    # Average Perceptron- The number of mistake per iteration
    #print(result2[1])

    # Average Perceptron- The number of accuracy per iteration
    # print(result2[2])

    # Average Perceptron- Final weight
    # print(result2[3])
    w2 = sum(result1[3], [])

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

    print("(Test) Plain Perceptron Binary:")
    pb = pba.PerceptronBinary(eta=1, n_iter=20)
    result3 = pb.fit(x2, y2, w1)

    # Perceptron- The number of mistake per iteration
    # print(result3[1])

    # Perceptron- The number of accuracy per iteration
    # print(result3[2])

    # Perceptron- Final weight
    # print(result3[3])

    print("(Test) Average Perceptron Binary:")
    apb = apba.AvgPerceptronBinary(n_iter=20)
    result4 = apb.fit(x2, y2, w2)

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
    print("Avg Perceptron Binary's TRAINING accuracy:", )
    print(result2[2])
    print("Final result")
    print("Perceptron Binary's TEST accuracy:", )
    print(result3[2])
    print("Final result")
    print("Avg Perceptron Binary's TEST accuracy:", )
    print(result4[2])

    # Accuracy curves
    plt.plot(result1[0], result1[2], label='Training_Plain')
    plt.plot(result2[0], result2[2], label='Training_Average')
    plt.plot(result3[0], result3[2], label='Test_Plain')
    plt.plot(result4[0], result4[2], label='Test_Average')
    plt.xlabel('Iteration')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
