import pandas as pd
import numpy as np

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pyplot as plt


def testKNN():
    # Creates list for creating testing for K
    kList = list(range(1, 50))
    # Using only odd numbers

    neighbours = list(filter(lambda x: x % 2 != 0, kList))
    # holds cross validations scores
    cv_scores = []

    for k in neighbours:
        knn = KNeighborsClassifier(n_neighbors=k)
        scores = cross_val_score(knn, x_train, y_train.values.ravel(), cv=10, scoring='accuracy')
        cv_scores.append(scores.mean())
    errors = [1 - x for x in cv_scores]

    optimal_k = neighbours[errors.index(min(errors))]
    print("The optimal number is %d" % optimal_k)

    # plot graph for best K
    plt.plot(neighbours, errors)
    plt.xlabel('Number')
    plt.ylabel('Errors')
    plt.show()


def printResults():

    # Finding the best value for K
    #testKNN()

    # 13 was most optimal
    knn = KNeighborsClassifier(n_neighbors=13)
    knn.fit(x_train, y_train.values.ravel())

    # Read in test data
    testData = pd.read_csv('./data/queries.txt', names=featureNames, na_values=['?'])

    # splitting cat and cont features
    cat_Test = testData[dt_cat]
    testFrame = testData.drop(dt_cat + ['id', 'balance', 'day', 'duration', 'previous', 'y'], axis=1)


    testcat_df = cat_Test.T.to_dict().values()
    testvec_cat_df = d_vectorizer.fit_transform(testcat_df)
    test_df = np.hstack((testFrame.as_matrix(), testvec_cat_df))

    test_pred = knn.predict(test_df)

    ids = testData['id'].ravel()


    withquotes = ["\"" + element + "\"" for element in test_pred]
    np.savetxt('predictions.txt', np.transpose([ids, withquotes]), fmt='%.18s', delimiter=',')



featureNames = []
with open('data/dataDescription.txt', 'r') as d:
    for line in d:
        if line[0].isdigit():
            items = line.split(' ')
            featureNames.append(items[2].strip().replace(':', ''))

df = pd.read_csv('./data/trainingset.txt', names=featureNames, na_values=['?'])


dt_cat = ['job', 'loan', 'marital', 'education', 'default', 'housing', 'contact', 'month', 'poutcome']
cat_dataFrame = df[dt_cat]

ContFrame = df.drop(dt_cat + ['id', 'balance', 'day', 'duration', 'previous', 'y'], axis=1)



#  Swaps '?' for 'NA'

cat_dataFrame.replace('?', 'NA')
cat_dataFrame.replace('unknown', 'NA')
cat_dataFrame.fillna('NA', inplace=True)



d_vectorizer = DictVectorizer(sparse=False)
cat_df = cat_dataFrame.T.to_dict().values()
vec_cat = d_vectorizer.fit_transform(cat_df)
train_df = np.hstack((ContFrame.as_matrix(), vec_cat))

target = df['y']
x_train, x_test, y_train, y_test = train_test_split(train_df, target, test_size=0.2, random_state=0)

printResults()