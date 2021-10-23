import pandas as pd
import PerceptronMulti as pma
import AvgPerceptronMulti as apma
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = pd.read_csv('./fashion-mnist_train.csv')

    label = dataset['label']
    pixels = dataset.iloc[:, 1:]

    x = pixels.values
    y = label.values

    print("(Training) Plain Perceptron Multi:")
    pm = pma.PerceptronMulti(eta=1, n_iter=20)
    result1 = pm.fit(x, y, 0)

    # Perceptron- The number of mistake per iteration
    # print(result1[1])

    # Perceptron- The number of accuracy per iteration
    # print(result1[2])

    # Perceptron- Final weight
    # print(result1[3])
    w1 = sum(result1[3], [])

    print("(Training) Average Perceptron Multi:")
    apm = apma.AvgPerceptronMulti(eta=1, n_iter=20)
    result2 = apm.fit(x, y, 0)

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

    x2 = tpixels.values
    y2 = tlabel.values

    print("(Test) Plain Perceptron Multi:")
    pm = pma.PerceptronMulti(eta=1, n_iter=20)
    result3 = pm.fit(x2, y2, w1)

    # Perceptron- The number of mistake per iteration
    # print(result3[1])

    # Perceptron- The number of accuracy per iteration
    # print(result3[2])

    # Perceptron- Final weight
    # print(result3[3])

    print("(Test) Average Perceptron Multi:")
    apm = apma.AvgPerceptronMulti(n_iter=20)
    result4 = apm.fit(x2, y2, w2)

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
