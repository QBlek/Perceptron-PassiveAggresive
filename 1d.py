import pandas as pd
import PerceptronBinary as pba
import PABinary as paba
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

    print("Perceptron Binary:")
    # For training - ranged data (100 - 10000)
    pb = pba.PerceptronBinary(eta=1, n_iter=20)
    # For test - unseen data (59900 - 50000)
    pbt = pba.PerceptronBinary(eta=1, n_iter=1)

    acc_num_sam = []
    for _ in range(2):
        acc_num_sam.append([])

    # for 100~10000
    num_sam = 101
    for _ in range(100):
        print(num_sam - 1, "' samples")
        training = pb.fit(x[1:num_sam], y[1:num_sam], 0)
        w = sum(training[3], [])
        test = pbt.fit(x[num_sam:], y[num_sam:], w)
        acc_num_sam[0].append(num_sam-1)
        acc_num_sam[1].append(test[2][-1])
        num_sam += 100

    # Perceptron-each acc
    print("Accuracy per training sample data")
    print(acc_num_sam)

    # Learning curves
    plt.plot(acc_num_sam[0], acc_num_sam[1], label='Perceptron')
    plt.xlabel('Num of training samples')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

