"""
COMP90055 Computing Project
Project title: "Machine learning for fertility prediction"

author: Yiming Zhang
"""

import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import MultinomialNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.svm import LinearSVC
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from pandas import ExcelWriter
from sklearn.model_selection import GridSearchCV
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
from sklearn.model_selection import learning_curve
from sklearn.model_selection import ShuffleSplit
import matplotlib.pyplot as plt


def delete_data(input_data):
    """
    Delete the target attributes
    :param input_data: data set
    :return: a data set which has no empty attributes
    """

    # delete the outcome attributes
    input_data.drop('Amen_ST60', axis=1, inplace=True)
    input_data.drop('AmenST36', axis=1, inplace=True)
    input_data.drop('AmenST48', axis=1, inplace=True)
    input_data.drop('target', axis=1, inplace=True)

    # delete the attributes related to "Amen"
    input_data.drop('Amen_ST12', axis=1, inplace=True)
    input_data.drop('Amen_ST24', axis=1, inplace=True)
    input_data.drop('Amen_ST3', axis=1, inplace=True)
    input_data.drop('Amen_ST6', axis=1, inplace=True)
    return input_data


def drop_data(input_data):
    """
       Drop the columns where all at least three elements are non-null
       :param input_data: data set
       :return: processed data set
    """
    input_data = input_data.dropna(axis=1, thresh=3)
    return input_data


def feature_scale(input_data):
    """
        Feature scaling
        :param input_data: data set
        :return: processed data set
    """
    scale = StandardScaler()
    input_data = scale.fit_transform(input_data)

    return input_data


def print_classification_report(classifier, train_data, target_data, num_validations):
    """
        print classification report
        :param: classifier, training data, target data, number of validations
        :return: print classification report
    """
    accuracy = cross_val_score(classifier, train_data, target_data, scoring='accuracy', cv=num_validations)
    print("Accuracy: " + str(round(accuracy.mean(), 4)))

    precision = cross_val_score(classifier, train_data, target_data, scoring='precision_weighted', cv=num_validations)
    print("Precision: " + str(round(precision.mean(), 4)))

    recall = cross_val_score(classifier, train_data, target_data, scoring='recall_weighted', cv=num_validations)
    print("Recall: " + str(round(recall.mean(), 4)))

    f1 = cross_val_score(classifier, train_data, target_data, scoring='f1_weighted', cv=num_validations)
    print("F1: " + str(round(f1.mean(), 4)))

    auc = cross_val_score(classifier, train_data, target_data, scoring='roc_auc', cv=num_validations)
    print("AUC: " + str(round(auc.mean(), 4)))


def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=None, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    return plt


if __name__ == "__main__":
    # read data from csv file
    data = pd.read_excel('dataSet.xlsx')
    df = pd.DataFrame(data)

    """
    Data pre-processing:
    1. Find target attribute in the data.
    2. Delete target attributes from the training data and testing data.
    3. If most of the values in a feature is null (less than 3 instances), then delete this feature.
    4. Convert categorical value in the feature to numerical value.
    5. If some of the value is missing in a feature then uses the mean to replace the missing values.
    6. Feature scaling.
    """

    print("------------------------------------")
    print("Data Pre Processing")
    print("------------------------------------")
    print("\n")

    # original data
    print("------------------------------------")
    print("Original data:")
    print("------------------------------------")
    print(df.shape)
    print("\n")

    # delete instances where all target features are empty

    df = df.dropna(subset=['Amen_ST60', 'AmenST36', 'AmenST48'], how='all')
    print("After drop empty targets:")
    print(df.shape)
    print("\n")

    # get target feature
    df['target'] = 0
    for index in df.index:
        if df.at[index, 'Amen_ST60'] == 1.0 or df.at[index, 'AmenST36'] == 1.0 or df.at[index, 'AmenST48'] == 1.0:
            df.at[index, 'target'] = 1
        else:
            df.at[index, 'target'] = 0

    target = df[['target']]
    print(target)

    # apply delete_data function to delete irrelevant features
    df = delete_data(df)
    print("------------------------------------")
    print("After delete data")
    print("------------------------------------")
    print(df.isnull().sum())
    print(df.shape)
    print(df)
    print("\n")

    # apply drop_data function to drop empty features
    df = drop_data(df)
    print("------------------------------------")
    print("After drop data")
    print("------------------------------------")
    print(df.isnull().sum())
    print(df.shape)
    print(df)
    print("\n")

        # convert FSH_T0 to float type
    df[['FSH_T0']] = df.FSH_T0.astype(float)

    # convert CT_Duration to float type
    for index in df.index:
        if df.at[index, 'CT_Duration'] == "> 64":
            df.at[index, 'CT_Duration'] = 65.0
        if df.at[index, 'CT_Duration'] == "<= 64":
            df.at[index, 'CT_Duration'] = 63.0
        if df.at[index, 'CT_Duration'] == " ":
            df.at[index, 'CT_Duration'] = 0.0
    df[['CT_Duration']] = df.CT_Duration.astype(float)

    # convert TD/wk to float type
    for index in df.index:
        if df.at[index, 'TD/wk'] == " ":
            df.at[index, 'TD/wk'] = 0.0
    df[['TD/wk']] = df[['TD/wk']].astype(float)

    # convert category feature to numerical feature
    df = pd.get_dummies(df)
    print("------------------------------------")
    print("convert category data")
    print("------------------------------------")
    print(df.shape)
    print(df.head())
    print("\n")

    # convert all string type to float type in order to feature scale
    cols = df.columns[df.dtypes.eq('object')]
    df[cols] = df[cols].apply(pd.to_numeric, errors='coerce')

    # Ignore UndefinedMetricWarning in precision, recall, F-score in some classifiers
    warnings.filterwarnings("ignore")

    # fill the empty data by filling null with its feature's mean value
    df.fillna(df.mean(), inplace=True)
    print("------------------------------------")
    print("After fill null")
    print("------------------------------------")
    print(df.isnull().sum())
    print("\n")
    print(df)
    print("\n")

    # df = pd.DataFrame(df).reset_index()
    # X is the training data, y is the target data
    X = df
    # convert column to row
    y = target.values.ravel()

    # feature selection: uses chi-square to select best k features
    selector = SelectKBest(score_func=chi2, k='all')
    # selector = SelectKBest(score_func=chi2, k=50)
    # selector = SelectKBest(score_func=chi2, k=100)
    # selector = SelectKBest(score_func=chi2, k=200)

    selector.fit(X, y)

    X_new = selector.transform(X)
    print("------------------------------------")
    print("After feature selection")
    print("------------------------------------")
    print(X_new.shape)
    print("\n")
    print(X_new)
    print("------------------------------------")
    print("Chi square score")
    print("------------------------------------")
    # scores
    scores = selector.scores_

    print(len(scores))
    print(len(X.columns))
    # attributes names of the k highest chi square scores
    vector_names = list(X.columns[selector.get_support(indices=True)])
    print(vector_names[:100])
    print(len(vector_names))

    chi2_scores = list(zip(X.columns, scores))
    chi2_scores.sort(reverse=True, key=lambda x: x[1])
    print(chi2_scores)
    pd.DataFrame(chi2_scores).to_excel('new_chi2.xlsx', header=False, index=False)
    print("\n")

    X_new = feature_scale(X_new)
    print("------------------------------------")
    print("After feature scale")
    print("------------------------------------")
    print(X_new.shape)
    print(X_new)
    print("\n")

    # save data to new file

    # writer = ExcelWriter('testing.xlsx')
    # pd.DataFrame(df).to_excel(writer, 'Sheet1')
    # writer.save()

    """
    Algorithms (10 fold cross validation)
    1. K-Nearest Neighbour (KNN)
    2. Logistic Regression
    3. Naive Bayes
    4. Support Vector Machine
    5. Random Forest
    6. Artificial Neural Network
    """

    print("------------------------------------------------")
    print("Algorithms")
    print("------------------------------------------------")
    print(X_new.shape)
    print("\n")

    """
    Algorithm1: K-Nearest Neighbour (KNN)
    
    """

    print("------------------------------------")
    print("Algorithm 1: KNN ")
    print("------------------------------------")

    # Use grid search CV method to find the optimal parameters: n_neighbor = 15 performs best
    """
    k_range = range(1, 20)
    weight_options = ['uniform', 'distance']
    para_grid = dict(n_neighbors=k_range, weights=weight_options)

    grid_knn = GridSearchCV(knn, para_grid, cv=10, scoring='accuracy')
    grid_knn.fit(X, y)

    # examine the best model
    print(grid_knn.best_score_)
    print(grid_knn.best_params_)
    print(grid_knn.best_estimator_)
    """

    # knn1 = KNeighborsClassifier(n_neighbors=5)
    # print(knn1)
    # print_classification_report(knn1, X_new, y, 10)
    # print("\n")

    knn2 = KNeighborsClassifier(n_neighbors=15)
    print(knn2)
    print_classification_report(knn2, X_new, y, 10)
    print("\n")

    # title1 = "Learning Curves (KNN 1)"
    # estimator1 = knn1
    cv = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # plot_learning_curve(estimator1, title1, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title2 = "Learning Curves (K Nearest Neighbour)"
    estimator2 = knn2
    plot_learning_curve(estimator2, title2, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    print("------------------------------------")
    print("Algorithm 2: Logistic Regression ")
    print("------------------------------------")

    """
    # Use grid search CV method to find the optimal parameters: 'lbfgs' solver performs best
    solver_options = ['lbfgs', 'liblinear']
    para_grid = dict(solver=solver_options)

    grid_lr = GridSearchCV(lr, para_grid, cv=10, scoring='accuracy')
    grid_lr.fit(X, y)

    # examine the best model
    print(grid_lr.best_score_)
    print(grid_lr.best_params_)
    print(grid_lr.best_estimator_)
    """

    lr1 = LogisticRegression(solver='lbfgs')
    print(lr1)
    print_classification_report(lr1, X_new, y, 10)
    print("\n")

    # lr2 = LogisticRegression(solver='liblinear')
    # print(lr2)
    # print_classification_report(lr2, X_new, y, 10)
    # print("\n")

    title3 = "Learning Curves (Logistic Regression)"
    estimator3 = lr1
    plot_learning_curve(estimator3, title3, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    # title4 = "Learning Curves (LR2)"
    # estimator4 = lr2
    # plot_learning_curve(estimator4, title4, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    print("------------------------------------")
    print("Algorithm 3: Naive Bayes ")
    print("------------------------------------")
    # nb = GaussianNB()
    # print(nb)
    # print_classification_report(nb, X_new, y, 10)
    # print("\n")

    nb2 = BernoulliNB()
    print(nb2)
    print_classification_report(nb2, X_new, y, 10)
    print("\n")

    # title5 = "Learning Curves (NB1)"
    # estimator5 = nb
    # cv1 = ShuffleSplit(n_splits=10, test_size=0.2, random_state=0)
    # plot_learning_curve(estimator5, title5, X_new, y, ylim=(0.3, 0.71), cv=cv, n_jobs=4)

    title6 = "Learning Curves (Naive Bayes)"
    estimator6 = nb2
    plot_learning_curve(estimator6, title6, X_new, y, ylim=(0.4, 0.91), cv=cv, n_jobs=4)

    print("------------------------------------")
    print("Algorithm 4: Support Vector Machine ")
    print("------------------------------------")
    
    svm1 = svm.SVC(kernel="linear")
    print(svm1)
    print_classification_report(svm1, X_new, y, 10)
    print("\n")

    # linear_svm = LinearSVC(dual=False, max_iter=2500)
    # print(linear_svm)
    # print_classification_report(linear_svm, X_new, y, 10)
    # print("\n")

    title7 = "Learning Curves (Support Vector Machine)"
    estimator7 = svm1
    plot_learning_curve(estimator7, title7, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    # title8 = "Learning Curves (SVM2)"
    # estimator8 = linear_svm
    # plot_learning_curve(estimator8, title8, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    print("------------------------------------")
    print("Algorithm 5: Random Forest")
    print("------------------------------------")
    # Instantiate model with 1000 decision trees
    # rf = RandomForestClassifier(n_estimators=1000)
    # print(rf)
    # print_classification_report(rf, X_new, y, 10)
    # print("\n")

    rf2 = RandomForestClassifier(n_estimators=200)
    print(rf2)
    print_classification_report(rf2, X_new, y, 10)
    print("\n")

    # title9 = "Learning Curves (RF1)"
    # estimator9 = rf
    # plot_learning_curve(estimator9, title9, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title10 = "Learning Curves (Random Forest)"
    estimator10 = rf2
    plot_learning_curve(estimator10, title10, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    """
    iter_options = [100, 200, 300, 400, 500, 1000]
    para_grid = dict(n_estimators=iter_options)

    grid_rf = GridSearchCV(rf, para_grid, cv=10, scoring='accuracy')
    grid_rf.fit(X, y)
    
    # examine the best model
    print(grid_rf.best_score_)
    print(grid_rf.best_params_)
    print(grid_rf.best_estimator_)
    
    """
    print("------------------------------------")
    print("Algorithm 6: Artificial Neural Network")
    print("------------------------------------")
    # neural network with three hidden layers( each 10 nodes) and number of iteration is 1000
    # mlp1 = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100), max_iter=10000)
    # print(mlp1)
    # print_classification_report(mlp1, X_new, y, 10)
    # print("\n")
    #
    # mlp2 = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100, 100), max_iter=10000)
    # print(mlp2)
    # print_classification_report(mlp2, X_new, y, 10)
    # print("\n")

    mlp3 = MLPClassifier(solver='adam', hidden_layer_sizes=(100, 100, 100, 100, 100), max_iter=10000)
    print(mlp3)
    print_classification_report(mlp3, X_new, y, 10)
    print("\n")

    # title11 = "Learning Curves (ANN1)"
    # estimator11 = mlp1
    # plot_learning_curve(estimator11, title11, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)
    #
    # title12 = "Learning Curves (ANN2)"
    # estimator12 = mlp2
    # plot_learning_curve(estimator12, title12, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    title13 = "Learning Curves (Artificial Neural Network)"
    estimator13 = mlp3
    plot_learning_curve(estimator13, title13, X_new, y, ylim=(0.7, 1.01), cv=cv, n_jobs=4)

    plt.show()

    """
    hidden_layers_options = [(100), (100,100), (100, 100, 100), (100, 100, 100, 100), (100, 100, 100, 100, 100)]
    para_grid = dict(hidden_layer_sizes=hidden_layers_options)

    grid_mlp = GridSearchCV(mlp, para_grid, cv=10, scoring='accuracy')
    grid_mlp.fit(X, y)

    # examine the best model
    print(grid_mlp.best_score_)
    print(grid_mlp.best_params_)
    print(grid_mlp.best_estimator_)
    """

