import pandas as pd
import PerceptronMulti as pma
import PAMulti as pama
import matplotlib.pyplot as plt

if __name__ == '__main__':

    dataset = pd.read_csv('./fashion-mnist_train.csv')

    label = dataset['label']
    pixels = dataset.iloc[:, 1:]

    label_val = label.values
    x = pixels.values
    y = label.values

    print("Perceptron Binary:")
    pm = pma.PerceptronMulti(eta=1, n_iter=50)
    result1 = pm.fit(x, y, 0)

    # Perceptron- The number of mistake per iteration
    # print(result1[1])

    # Perceptron- The number of accuracy per iteration
    # print(result1[2])

    # Perceptron- Final weight
    # print(result1[3])

    print("PA Binary:")
    pam = pama.PAMulti(n_iter=50)
    result2 = pam.fit(x, y, 0)

    # PA- The number of mistake per iteration
    # print(result2[1])

    # PA - The number of accuracy per iteration
    # print(result2[2])

    # PA - Final weight
    # print(result2[3])

    print("")
    print("Final result")
    print("Perceptron Binary's the number of mistake:", )
    print(result1[1])
    print("")
    print("Final result")
    print("PA Binary's the number of mistake:", )
    print(result2[1])

    # Learning curves
    plt.plot(result1[0], result1[1], label='Perceptron')
    plt.plot(result2[0], result2[1], label='PA')
    plt.xlabel('Iteration')
    plt.ylabel('Num of mistake')
    plt.legend()
    plt.show()


