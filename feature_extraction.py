import pandas as pd
import numpy as np
import re
from sklearn import preprocessing, model_selection, linear_model, metrics, svm, tree, ensemble
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline


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


def get_jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersect = set1.intersection(set2)
    return float(len(intersect)) / (len(set1) + len(set2) - len(intersect))


def mean_ce(lis):
    return sum(lis) / len(lis)


def run_logistic_regression(train_x, train_y, test_x, test_y):
    lr_model = linear_model.LogisticRegression(C=1, solver="saga")
    lr_model.fit(train_x, train_y)
    lr_preds = lr_model.predict(test_x)

    lr_corr = []
    for i in range(len(lr_preds)):
        if lr_preds[i] == test_y[i]:
            pass
            # lr_corr.append((1, test_y[i]))
        else:
            lr_corr.append(test_y[i])

    # print(np.unique(lr_corr, return_counts=True))

    lr_acc = lr_model.score(test_x, test_y)
    lr_f1 = metrics.f1_score(test_y, lr_preds, average='weighted', labels=[0, 1, 2])
    lr_kappa = metrics.cohen_kappa_score(test_y, lr_preds)

    return "LR model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(lr_acc, lr_f1, lr_kappa)


def run_svm(train_x, train_y, test_x, test_y):
    svm_c = svm.SVC(C=1, kernel='rbf', gamma='scale').fit(train_x, train_y)
    svm_preds = svm_c.predict(test_x)

    svm_corr = []
    for i in range(len(svm_preds)):
        if svm_preds[i] == test_y[i]:
            pass
            # lr_corr.append((1, test_y[i]))
        else:
            svm_corr.append(test_y[i])

    # print(np.unique(svm_corr, return_counts=True))

    svm_acc = svm_c.score(test_x, test_y)
    svm_f1 = metrics.f1_score(test_y, svm_preds, average='weighted', labels=[0, 1, 2])
    svm_kappa = metrics.cohen_kappa_score(test_y, svm_preds)

    return "SVM model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(svm_acc, svm_f1, svm_kappa)


def run_dt(train_x, train_y, test_x, test_y):
    dt_model = tree.DecisionTreeClassifier().fit(train_x, train_y)
    dt_preds = dt_model.predict(test_x)

    dt_corr = []
    for i in range(len(dt_preds)):
        if dt_preds[i] == test_y[i]:
            pass
        else:
            dt_corr.append(test_y[i])

    # print(np.unique(dt_corr, return_counts=True))

    dt_acc = dt_model.score(test_x, test_y)
    dt_f1 = metrics.f1_score(test_y, dt_preds, average='weighted', labels=[0, 1, 2])
    dt_kappa = metrics.cohen_kappa_score(test_y, dt_preds)

    return "DT model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(dt_acc, dt_f1, dt_kappa)


def run_rf(train_x, train_y, test_x, test_y):
    # rf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy')
    # rf.fit(train_x, train_y)
    # rf_preds = rf.predict(test_x)
    #
    # rf_corr = []
    # mislabels = []
    # for i in range(len(rf_preds)):
    #     if rf_preds[i] == test_y[i]:
    #         pass
    #     else:
    #         rf_corr.append(test_y[i])
    #         if test_y[i] == 1:
    #             mislabels.append(rf_preds[i])
    #
    # print(np.unique(test_y, return_counts=True))
    # print(np.unique(rf_corr, return_counts=True))
    # print(np.unique(mislabels, return_counts=True))
    #
    # rf_acc = rf.score(test_x, test_y)
    # rf_f1 = metrics.f1_score(test_y, rf_preds, average='weighted', labels=[0, 1, 2])
    # rf_kappa = metrics.cohen_kappa_score(test_y, rf_preds)

    pipeline = Pipeline([('over', SMOTE("minority")), ('model', ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy'))])
    scores = model_selection.cross_val_score(pipeline, train_x, train_y)
    print(scores)
    # return "RF model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(rf_acc, rf_f1, rf_kappa)


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

    data_new.insert(12, 'My_CE', 0, True)

    # Simplifying CE labels
    data_new.loc[data_new['CE_label'] == 'C1', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'C2', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'I', 'My_CE'] = 'I'
    data_new.loc[data_new['CE_label'] == 'A1', 'My_CE'] = 'A'

    print(np.unique(data_new["My_CE"], return_counts=True))

    # Preprocess relevant text: comment text and the highlighted area from source material
    data_new['text_without_emoji'] = prepross(data_new['text_without_emoji'])
    data_new['highlighted+text'] = prepross(data_new['highlighted+text'])

    # Jaccard similarity between the comment text and the highlighted text
    similarities = []
    for i in range(len(data_new)):
        similarities.append(get_jaccard(data_new['text_without_emoji'][i], data_new['highlighted+text'][i]))

    data_new['jaccard_sims'] = similarities

    # Add whether the post is a reply or not
    is_reply = []
    for i in range(len(data_new)):
        if data_new['parent_id'][i] > 0:
            is_reply.append(1)
        else:
            is_reply.append(0)

    data_new['is_reply'] = is_reply

    # Add Nota Bene tag data
    tag_names = ['#idea', '#question', '#help', '#frustrated', '#interested', '#confused', '#useful', '#curious']

    for n in tag_names:
        temp = []
        for i in range(len(data_new)):
            if int(data_new[n][i]) == 1:
                temp.append(1)
            else:
                temp.append(0)

        data_new[n] = temp

    # Add comment length
    lengths = []
    for i in range(len(data_new)):
        length = len(data_new["text_without_emoji"][i])
        if length > 750:
            lengths.append(1.0)
        else:
            lengths.append(length / 750)

    data_new['length'] = lengths

    data_x = data_new.drop(columns=['CE_label', 'My_CE'])
    data_x.drop(columns=['text', 'text_without_emoji', 'highlighted+text', 'parent_id'], inplace=True)
    labels = data_new['My_CE']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    most_freq_label = float(np.bincount(labels).argmax())

    # smote = SMOTE("minority")
    # samp_x, samp_y = smote.fit_sample(data_x, labels)
    train_x, test_x, train_y, test_y = model_selection.train_test_split(data_x, labels, test_size=0.15)
    # train_x, test_x, train_y, test_y = model_selection.train_test_split(data_x, labels, test_size=0.15)

    # train_x, train_y = smote.fit_sample(train_x, train_y)

    train_x.reset_index(inplace=True)
    test_x.reset_index(inplace=True)

    # create a count vectorizer object
    # count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', min_df=100)
    # text_train = count_vect.fit_transform(train_x['text_without_emoji'])
    # text_test = count_vect.transform(test_x['text_without_emoji'])

    # Dataframe where columns show existance of each term for all posts
    # features_train = pd.DataFrame(text_train.todense(), columns=count_vect.get_feature_names())
    # features_test = pd.DataFrame(text_test.todense(), columns=count_vect.get_feature_names())

    # Average CE of that student/author
    authors_dict = {}
    for i in range(len(train_x)):
        aid = str(train_x['author_id'][i])
        ce = int(str(train_y[i]))
        try:
            authors_dict[aid].append(ce)
        except KeyError:
            authors_dict[aid] = [ce]

    # Dict of most common CE for each author
    for k in authors_dict:
        authors_dict[k] = mean_ce(authors_dict[k])

    avg_ce_train = []
    avg_ce_test = []
    for i in range(len(train_x)):
        try:
            avg_ce_train.append(authors_dict[str(train_x['author_id'][i])])
        except KeyError:
            avg_ce_train.append(most_freq_label)
    for i in range(len(test_x)):
        try:
            avg_ce_test.append(authors_dict[str(test_x['author_id'][i])])
        except KeyError:
            avg_ce_test.append(most_freq_label)

    features_train = pd.DataFrame()
    features_test = pd.DataFrame()

    # Add to feature space
    features_train['avg_ce'] = avg_ce_train
    features_test['avg_ce'] = avg_ce_test

    features_train['is_reply'] = train_x['is_reply']
    features_test['is_reply'] = test_x['is_reply']

    features_train['jaccard_sims'] = train_x['jaccard_sims']
    features_test['jaccard_sims'] = test_x['jaccard_sims']

    features_train['length'] = train_x['length']
    features_test['length'] = test_x['length']

    features_train['Confusion_label'] = train_x['Confusion_label']
    features_test['Confusion_label'] = test_x['Confusion_label']

    features_train['#useful'] = train_x['#useful']
    features_test['#useful'] = test_x['#useful']
    features_train['#idea'] = train_x['#idea']
    features_test['#idea'] = test_x['#idea']
    features_train['#question'] = train_x['#question']
    features_test['#question'] = test_x['#question']
    features_train['#interested'] = train_x['#interested']
    features_test['#interested'] = test_x['#interested']
    features_train['#curious'] = train_x['#curious']
    features_test['#curious'] = test_x['#curious']

    # features_train['#help'] = train_x['#help']
    # features_test['#help'] = test_x['#help']
    # features_train['#confused'] = train_x['#confused']
    # features_test['#confused'] = test_x['#confused']
    # features_train['#frustrated'] = train_x['#frustrated']
    # features_test['#frustrated'] = test_x['#frustrated']

    lr_out = run_logistic_regression(features_train, train_y, features_test, test_y)
    svm_out = run_svm(features_train, train_y, features_test, test_y)
    dt_out = run_dt(features_train, train_y, features_test, test_y)
    run_rf(features_train, train_y, features_test, test_y)

    print(lr_out)
    print(svm_out)
    print(dt_out)
    # print(rf_out)


if __name__ == '__main__':
    main()
