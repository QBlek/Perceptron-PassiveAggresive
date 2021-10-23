import pandas as pd
import PerceptronMulti as pma
import PAMulti as pama
import matplotlib.pyplot as plt

if __name__ == '__main__':

    training_set = pd.read_csv('./fashion-mnist_train.csv')

    nlabel = training_set['label']
    npixels = training_set.iloc[:, 1:]

    x = npixels.values
    y = nlabel.values

    print("(Training) Perceptron Multi:")
    pm = pma.PerceptronMulti(eta=1, n_iter=20)
    result1 = pm.fit(x, y, 0)

    # Perceptron- The number of mistake per iteration
    # print(result1[1])

    # Perceptron- The number of accuracy per iteration
    # print(result1[2])

    # Perceptron- Final weight
    # print(result1[3])
    w1 = sum(result1[3], [])


    print("(Training) PA Multi:")
    pam = pama.PAMulti(n_iter=20)
    result2 = pam.fit(x, y, 0)

    # PA- The number of mistake per iteration
    # print(result2[1])

    # PA - The number of accuracy per iteration
    # print(result2[2])

    # PA - Final weight
    # print(result2[3])
    w2 = sum(result2[3], [])

    test_set = pd.read_csv('fashion-mnist_test.csv')

    tlabel = test_set['label']
    tpixels = test_set.iloc[:, 1:]

    x2 = tpixels.values
    y2 = tlabel.values

    print("(Test) Perceptron Multi:")
    pm = pma.PerceptronMulti(n_iter=20)
    result3 = pm.fit(x2, y2, w1)

    # Perceptron- The number of mistake per iteration
    # print(result3[1])

    # Perceptron- The number of accuracy per iteration
    # print(result3[2])

    # Perceptron- Final weight
    # print(result3[3])

    print("(Test) PA Multi:")
    pam = pama.PAMulti(n_iter=20)
    result4 = pam.fit(x2, y2, w2)

    # PA- The number of mistake per iteration
    # print(result4[1])

    # PA - The number of accuracy per iteration
    # print(result4[2])

    # PA - Final weight
    # print(result4[3])

    print("")
    print("Final result")
    print("Perceptron Multi's TRAINING accuracy:", )
    print(result1[2])
    print("")
    print("Final result")
    print("PA Multi's TRAINING accuracy:", )
    print(result2[2])
    print("Final result")
    print("Perceptron Multi's TEST accuracy:", )
    print(result3[2])
    print("Final result")
    print("PA Multi's TEST accuracy:", )
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
