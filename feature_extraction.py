import pandas as pd
# import numpy as np
import re
from sklearn import preprocessing, model_selection, linear_model, metrics, svm, tree, ensemble
from sklearn.feature_extraction.text import CountVectorizer
# from nltk.corpus import stopwords
# from nltk.stem import PorterStemmer


def prepross(text_new):
    # ps = PorterStemmer()
    cpy = text_new.copy()
    
    for i in range(len(cpy)):
        # Tokenisation
        split1 = re.split('([\"\'?!#])', cpy.iloc[i])

        tks = []
        for j in split1:
            temp = re.split('[^a-zA-Z0-9\"\'?!#-]', j)
            for k in temp:
                tks.append(k)

        tokens = [t for t in tks if t != '']

        # Case Folding
        # lower_toks = list(map(lambda x: x.lower(), tokens))

        # Stopping
        # stop_words = set(stopwords.words('english'))
        # terms = []
        #
        # for w in lower_toks:
        #     if w not in stop_words:
        #     terms.append(w)

        # Normalisation
        # for l in range(len(tokens)):
        #     tokens[l] = ps.stem(tokens[l])
        #
        # cpy.iloc[i] = " ".join(tokens)

        cpy.iloc[i] = " ".join(tokens)

    return cpy


def run_logistic_regression(xtrain_count, train_y, xvalid_count, valid_y):
    # Logistic Regression
    lr_model = linear_model.LogisticRegression(C=1, solver='saga', random_state=100, multi_class='multinomial')
    lr_model.fit(xtrain_count, train_y)
    lr_preds = lr_model.predict(xvalid_count)

    lr_acc = lr_model.score(xvalid_count, valid_y)
    lr_f1 = metrics.f1_score(valid_y, lr_preds, average='weighted', labels=[0, 1, 2])
    lr_kappa = metrics.cohen_kappa_score(valid_y, lr_preds)

    # lr_scores = np.array([lr_acc, lr_f1, lr_kappa])
    return "LR model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(lr_acc, lr_f1, lr_kappa)


def run_svm(xtrain_count, train_y, xvalid_count, valid_y):
    # SVM
    svm_c = svm.SVC(C=1, kernel='rbf', random_state=100, gamma='scale').fit(xtrain_count, train_y)
    svm_preds = svm_c.predict(xvalid_count)

    svm_acc = svm_c.score(xvalid_count, valid_y)
    svm_f1 = metrics.f1_score(valid_y, svm_preds, average='weighted', labels=[0, 1, 2])
    svm_kappa = metrics.cohen_kappa_score(valid_y, svm_preds)

    # svm_scores = np.array([svm_acc, svm_f1, svm_kappa])
    return "SVM model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(svm_acc, svm_f1, svm_kappa)


def run_dt(xtrain_count, train_y, xvalid_count, valid_y):
    dt_model = tree.DecisionTreeClassifier(random_state=100).fit(xtrain_count, train_y)
    dt_preds = dt_model.predict(xvalid_count)

    dt_acc = dt_model.score(xvalid_count, valid_y)
    dt_f1 = metrics.f1_score(valid_y, dt_preds, average='weighted', labels=[0, 1, 2])
    dt_kappa = metrics.cohen_kappa_score(valid_y, dt_preds)

    # dt_scores = np.array([dt_acc, dt_f1, dt_kappa])
    return "DT model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(dt_acc, dt_f1, dt_kappa)


def run_rf(xtrain_count, train_y, xvalid_count, valid_y):
    rf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, random_state=100, criterion='entropy')
    rf.fit(xtrain_count, train_y)
    rf_preds = rf.predict(xvalid_count)
    print(rf_preds)

    rf_acc = rf.score(xvalid_count, valid_y)
    rf_f1 = metrics.f1_score(valid_y, rf_preds, average='weighted', labels=[0, 1, 2])
    rf_kappa = metrics.cohen_kappa_score(valid_y, rf_preds)

    # rf_scores = np.array([rf_acc, rf_f1, rf_kappa])
    return "RF model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(rf_acc, rf_f1, rf_kappa)


def main():
    data_raw = pd.read_excel('data.xlsx', index_col=0)

    data_trim = data_raw[['text', 'text_without_emoji', 'parent_id', '#idea', '#question', '#help', '#frustrated',
                          '#interested', '#confused', '#useful', '#curious', 'Paragraph_Text', 'CE_label',
                          'Confusion_label']]

    for i in range(len(data_trim)):
        if not isinstance(data_trim['text_without_emoji'][i], str):
            data_trim = data_trim.drop(i)

    # Remove entries with only hashtags
    data_trim.reset_index(drop=True, inplace=True)

    data_trim.insert(12, 'My_CE', 0, True)

    data_trim.loc[data_trim['CE_label'] == 'C1', 'My_CE'] = 'C'
    data_trim.loc[data_trim['CE_label'] == 'C2', 'My_CE'] = 'C'
    data_trim.loc[data_trim['CE_label'] == 'I', 'My_CE'] = 'I'
    data_trim.loc[data_trim['CE_label'] == 'A1', 'My_CE'] = 'A'

    comment_text = prepross(data_trim['text_without_emoji'])
    labels = data_trim['My_CE']

    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(comment_text, labels, test_size=0.15)

    # label encode the target variable
    encoder = preprocessing.LabelEncoder()
    encoder.fit(train_y)
    train_y = encoder.transform(train_y)
    valid_y = encoder.transform(valid_y)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', min_df=100)
    count_vect.fit(comment_text)

    # transform the training and validation data using count vectorizer object
    xtrain_count = count_vect.transform(train_x)
    xvalid_count = count_vect.transform(valid_x)

    lr_out = run_logistic_regression(xtrain_count, train_y, xvalid_count, valid_y)
    svm_out = run_svm(xtrain_count, train_y, xvalid_count, valid_y)
    dt_out = run_dt(xtrain_count, train_y, xvalid_count, valid_y)
    rf_out = run_rf(xtrain_count, train_y, xvalid_count, valid_y)

    print(lr_out)
    print(svm_out)
    print(dt_out)
    print(rf_out)


if __name__ == '__main__':
    main()
