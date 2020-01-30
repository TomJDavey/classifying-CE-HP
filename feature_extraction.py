import pandas as pd
import numpy as np
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


def get_jaccard(s1, s2):
    set1 = set(s1.split())
    set2 = set(s2.split())
    intersect = set1.intersection(set2)
    return float(len(intersect)) / (len(set1) + len(set2) - len(intersect))


def most_freq_element(lis):
    return max(set(lis), key=lis.count)


def run_logistic_regression(train_x, train_y, test_x, test_y):
    lr_model = linear_model.LogisticRegression(C=1, solver='saga', random_state=100, multi_class='multinomial')
    lr_model.fit(train_x, train_y)
    lr_preds = lr_model.predict(test_x)

    lr_acc = lr_model.score(test_x, test_y)
    lr_f1 = metrics.f1_score(test_y, lr_preds, average='weighted', labels=[0, 1, 2])
    lr_kappa = metrics.cohen_kappa_score(test_y, lr_preds)

    return "LR model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(lr_acc, lr_f1, lr_kappa)


def run_svm(train_x, train_y, test_x, test_y):
    svm_c = svm.SVC(C=1, kernel='rbf', random_state=100, gamma='scale').fit(train_x, train_y)
    svm_preds = svm_c.predict(test_x)

    svm_acc = svm_c.score(test_x, test_y)
    svm_f1 = metrics.f1_score(test_y, svm_preds, average='weighted', labels=[0, 1, 2])
    svm_kappa = metrics.cohen_kappa_score(test_y, svm_preds)

    return "SVM model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(svm_acc, svm_f1, svm_kappa)


def run_dt(train_x, train_y, test_x, test_y):
    dt_model = tree.DecisionTreeClassifier(random_state=100).fit(train_x, train_y)
    dt_preds = dt_model.predict(test_x)

    dt_acc = dt_model.score(test_x, test_y)
    dt_f1 = metrics.f1_score(test_y, dt_preds, average='weighted', labels=[0, 1, 2])
    dt_kappa = metrics.cohen_kappa_score(test_y, dt_preds)

    return "DT model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(dt_acc, dt_f1, dt_kappa)


def run_rf(train_x, train_y, test_x, test_y):
    rf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, random_state=100, criterion='entropy')
    rf.fit(train_x, train_y)
    rf_preds = rf.predict(test_x)

    rf_acc = rf.score(test_x, test_y)
    rf_f1 = metrics.f1_score(test_y, rf_preds, average='weighted', labels=[0, 1, 2])
    rf_kappa = metrics.cohen_kappa_score(test_y, rf_preds)

    return "RF model:   Accuracy: {0:2.3f}, F1 score: {1:2.3f}, Kappa: {2:2.3f}".format(rf_acc, rf_f1, rf_kappa)


def main():
    data_raw = pd.read_excel('data.xlsx', index_col=0)

    data_new = data_raw[['text', 'text_without_emoji', 'author_id', 'parent_id', '#idea', '#question', '#help', '#frustrated',
                          '#interested', '#confused', '#useful', '#curious', 'highlighted+text', 'Paragraph_Text', 'CE_label',
                          'Confusion_label']]

    for i in range(len(data_new)):
        if not isinstance(data_new['text_without_emoji'][i], str) or not isinstance(data_new['highlighted+text'][i], str):
            data_new = data_new.drop(i)

    # Remove entries with only hashtags
    data_new.reset_index(drop=True, inplace=True)

    data_new.insert(12, 'My_CE', 0, True)

    data_new.loc[data_new['CE_label'] == 'C1', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'C2', 'My_CE'] = 'C'
    data_new.loc[data_new['CE_label'] == 'I', 'My_CE'] = 'I'
    data_new.loc[data_new['CE_label'] == 'A1', 'My_CE'] = 'A'

    data_new['text_without_emoji'] = prepross(data_new['text_without_emoji'])
    data_new['highlighted+text'] = prepross(data_new['highlighted+text'])

    # Comment text concatenated with highlighted text for each post
    # all_text = data_new['text_without_emoji'].map(str) + " " + data_new['highlighted+text']

    # Jaccard similarity between the comment text and the highlighted text
    similarities = []
    for i in range(len(data_new)):
        similarities.append(get_jaccard(data_new['text_without_emoji'][i], data_new['highlighted+text'][i]))

    # Add whether the post is a reply or not
    is_reply = []
    for i in range(len(data_new)):
        if data_new['parent_id'][i] > 0:
            is_reply.append(1)
        else:
            is_reply.append(0)

    data_X = data_new.drop(columns=['CE_label', 'My_CE'])
    labels = data_new['My_CE']
    encoder = preprocessing.LabelEncoder()
    encoder.fit(labels)
    labels = encoder.transform(labels)
    most_freq_label = np.bincount(labels).argmax()
    # data_X['Jaccard sims'] = similarities
    data_X['is_reply'] = is_reply

    train_X, test_X, train_y, test_y = model_selection.train_test_split(data_X, labels, test_size=0.15)

    train_X.reset_index(inplace=True)
    test_X.reset_index(inplace=True)

    # create a count vectorizer object
    count_vect = CountVectorizer(analyzer='word', token_pattern=r'\w{1,}', min_df=100)
    text_train = count_vect.fit_transform(train_X['text_without_emoji'])
    text_test = count_vect.transform(test_X['text_without_emoji'])

    # Dataframe where columns show existance of each term for all posts
    features_train = pd.DataFrame(text_train.todense(), columns=count_vect.get_feature_names())
    features_test = pd.DataFrame(text_test.todense(), columns=count_vect.get_feature_names())
    # features['Average CE of author'] = avg_ce

    # Average CE of that student/author
    authors_dict = {}
    for i in range(len(train_X)):
        id = str(train_X['author_id'][i])
        ce = str(train_y[i])
        try:
            authors_dict[id].append(ce)
        except KeyError:
            authors_dict[id] = [ce]

    # Dict of most common CE for each author
    for k in authors_dict:
        authors_dict[k] = most_freq_element(authors_dict[k])

    avg_ce_train = []
    avg_ce_test = []
    for i in range(len(train_X)):
        try:
            avg_ce_train.append(authors_dict[str(train_X['author_id'][i])])
        except KeyError:

            avg_ce_train.append(most_freq_label)
    for i in range(len(test_X)):
        try:
            avg_ce_test.append(authors_dict[str(test_X['author_id'][i])])
        except KeyError:
            avg_ce_test.append(most_freq_label)

    # Add to feature space
    features_train['avg_ce'] = avg_ce_train
    features_test['avg_ce'] = avg_ce_test

    features_train['is_reply'] = train_X['is_reply']
    features_test['is_reply'] = test_X['is_reply']

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
