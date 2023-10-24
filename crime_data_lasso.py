if __name__ == "__main__":
    from ISTA import train  # type: ignore
else:
    from .ISTA import train

import matplotlib.pyplot as plt
import numpy as np


from utils import load_dataset, problem

def MSE(X,y):
    return(np.mean((X-y)**2))

@problem.tag("hw2-A", start_line=3)
def main():
    # df_train and df_test are pandas dataframes.
    # Make sure you split them into observations and targets
    df_train, df_test = load_dataset("crime")

    X_train = df_train.iloc[:, 1:].values
    y_train = df_train.iloc[:, 0].values
    X_test = df_test.iloc[:, 1:].values
    y_test = df_test.iloc[:, 0].values

    X_train = (X_train - X_train.mean(axis=0)) / X_train.std(axis=0)
    X_test = (X_test - X_train.mean(axis=0)) / X_train.std(axis=0)

    feature_list = ['agePct12t29', 'pctWSocSec', 'pctUrban', 'agePct65up', 'householdsize']
    feature_ind_list = df_train.columns.get_indexer(feature_list)-1

    _lambda_max = 2 * np.max(np.abs(X_train.T @ (y_train - np.mean(y_train))))
    _lambda = _lambda_max
    _lambda_list = []
    count = []
    train_MSE = []
    test_MSE = []
    coefficients = []

    while _lambda > 0.01:
        _lambda_list.append(_lambda)
        weight_train, bias_train = train(X_train, y_train, _lambda)
        count.append(np.sum(np.isclose(weight_train, 0) == False))
        train_MSE.append(MSE(X_train @ weight_train + bias_train, y_train))
        test_MSE.append(MSE(X_test @ weight_train + bias_train, y_test))
        coefficients.append([weight_train[j] for j in feature_ind_list])
        _lambda = _lambda / 2


    weight_train, bias_train = train(X_train, y_train, 30)
    max_idx = weight_train.argmax()
    max_val = weight_train[max_idx]
    max_name = df_train.columns[max_idx+1]
    min_idx = weight_train.argmin()
    min_val = weight_train[min_idx]
    min_name = df_train.columns[min_idx + 1]

    print(f"Max value: {max_val} for {max_name}")
    print(f"Min value: {min_val} for {min_name}")


    fig, axs = plt.subplots(nrows=3, ncols=1, figsize=(8, 12))

    # nonzero weights
    axs[0].set_title("Number of Nonzero Weights")
    axs[0].plot(_lambda_list, count, label="Nonzero Weights")
    axs[0].set_xlabel("Lambda")
    axs[0].set_ylabel("Count")
    axs[0].set_xscale('log')
    axs[0].set_ylim([0, X_train.shape[1]])

    # coefficient
    axs[1].set_title("Coefficient Paths")
    for i in range(len(feature_ind_list)):
        axs[1].plot(_lambda_list, [coef[i] for coef in coefficients], label=f"{feature_list[i]}")
    axs[1].set_xlabel("Lambda")
    axs[1].set_ylabel("Coefficients")
    axs[1].legend()
    axs[1].set_xscale('log')

    # MSE
    axs[2].set_title("Mean Squared Error")
    axs[2].plot(_lambda_list, train_MSE, label="Training MSE")
    axs[2].plot(_lambda_list, test_MSE, label="Test MSE")
    axs[2].set_xlabel("Lambda")
    axs[2].set_ylabel("MSE")
    axs[2].set_xscale('log')
    axs[2].legend()

    fig.subplots_adjust(wspace=0.4, hspace=0.3, bottom=0.1, top=0.9, right=0.9)
    #plt.savefig('6defplots.pdf')
    plt.show()


if __name__ == "__main__":
    main()
