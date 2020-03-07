import re
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn import preprocessing, model_selection, linear_model, metrics, svm, tree, ensemble


def prepross(text_new):
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

        cpy.iloc[i] = " ".join(tokens)

    return cpy


def run_logistic_regression(train_x, train_y, test_x, test_y):
    lr_model = linear_model.LogisticRegression(C=1, solver="saga")
    lr_model.fit(train_x, train_y)
    lr_preds = lr_model.predict(test_x)

    lr_acc = lr_model.score(test_x, test_y)
    lr_f1 = metrics.f1_score(test_y, lr_preds, average='weighted', labels=[0, 1, 2])
    lr_kappa = metrics.cohen_kappa_score(test_y, lr_preds)

    return "LR model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(lr_acc, lr_f1, lr_kappa)


def run_svm(train_x, train_y, test_x, test_y):
    svm_c = svm.SVC(C=1, kernel='rbf', gamma='scale').fit(train_x, train_y)
    svm_preds = svm_c.predict(test_x)

    svm_acc = svm_c.score(test_x, test_y)
    svm_f1 = metrics.f1_score(test_y, svm_preds, average='weighted', labels=[0, 1, 2])
    svm_kappa = metrics.cohen_kappa_score(test_y, svm_preds)

    return "SVM model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(svm_acc, svm_f1, svm_kappa)


def run_dt(train_x, train_y, test_x, test_y):
    dt_model = tree.DecisionTreeClassifier().fit(train_x, train_y)
    dt_preds = dt_model.predict(test_x)

    dt_acc = dt_model.score(test_x, test_y)
    dt_f1 = metrics.f1_score(test_y, dt_preds, average='weighted', labels=[0, 1, 2])
    dt_kappa = metrics.cohen_kappa_score(test_y, dt_preds)

    return "DT model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(dt_acc, dt_f1, dt_kappa)


def run_rf(train_x, train_y, test_x, test_y):
    rf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy')
    rf.fit(train_x, train_y)
    rf_preds = rf.predict(test_x)

    rf_corr = []
    mislabels = []
    for i in range(len(rf_preds)):
        if rf_preds[i] == test_y[i]:
            pass
        else:
            rf_corr.append(test_y[i])
            if test_y[i] == 1:
                mislabels.append(rf_preds[i])

    rf_acc = rf.score(test_x, test_y)
    rf_f1 = metrics.f1_score(test_y, rf_preds, average='weighted', labels=[0, 1, 2])
    rf_kappa = metrics.cohen_kappa_score(test_y, rf_preds)

    return "RF model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(rf_acc, rf_f1, rf_kappa)


def main():
    data_raw = pd.read_excel('data.xlsx', index_col=0)

    data_new = data_raw[
        ['text', 'text_without_emoji', 'author_id', 'parent_id', '#idea', '#question', '#help', '#frustrated',
         '#interested', '#confused', '#useful', '#curious', 'highlighted+text', 'CE_label', 'Confusion_label']]

    for i in range(len(data_new)):
        if not isinstance(data_new['text_without_emoji'][i], str) or not isinstance(data_new['highlighted+text'][i],
                                                                                    str):
            data_new = data_new.drop(i)

    # Remove entries with only hashtags
    data_new.reset_index(drop=True, inplace=True)

    # Simplifying CE labels
    data_new.insert(12, 'My_CE', 0, True)
    data_new.loc[data_new['CE_label'] == 'C1', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'C2', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'I', 'My_CE'] = 'I'
    data_new.loc[data_new['CE_label'] == 'A1', 'My_CE'] = 'A'

    # Preprocess relevant text: comment text and the highlighted area from source material
    data_new['text_without_emoji'] = prepross(data_new['text_without_emoji'])
    data_new['highlighted+text'] = prepross(data_new['highlighted+text'])

    data_x = data_new.drop(columns=['CE_label', 'My_CE'])
    data_x.drop(columns=['text', 'text_without_emoji', 'highlighted+text', 'parent_id'], inplace=True)
    labels = data_new['My_CE']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)

    train_x, test_x, train_y, test_y = model_selection.train_test_split(data_x, labels, test_size=0.15)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', min_df=100)
    text_train = count_vect.fit_transform(train_x['text_without_emoji'])
    text_test = count_vect.transform(test_x['text_without_emoji'])

    # Dataframe where columns show existance of each term for all posts
    features_train = pd.DataFrame(text_train.todense(), columns=count_vect.get_feature_names())
    features_test = pd.DataFrame(text_test.todense(), columns=count_vect.get_feature_names())

    lr_out = run_logistic_regression(features_train, train_y, features_test, test_y)
    svm_out = run_svm(features_train, train_y, features_test, test_y)
    dt_out = run_dt(features_train, train_y, features_test, test_y)
    rf_out = run_rf(features_train, train_y, features_test, test_y)

    print(lr_out)
    print(svm_out)
    print(dt_out)
    print(rf_out)


if __name__ == '__main__':
    main()
