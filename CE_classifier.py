import pandas as pd
import numpy as np
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline
from sklearn import preprocessing, model_selection, ensemble
from feature_extraction import prepross, get_jaccard, mean_ce


def run_rf(train_x, train_y):
    rf = ensemble.RandomForestClassifier(n_estimators=200, max_depth=4, criterion='entropy')
    pipeline = Pipeline([('over', SMOTE("minority")), ('model', rf)])
    scores = model_selection.cross_val_score(pipeline, train_x, train_y)

    print(np.mean(scores))


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

    # train_x, test_x, train_y, test_y = model_selection.train_test_split(data_x, labels, test_size=0.15)
    #
    # train_x.reset_index(inplace=True)
    # test_x.reset_index(inplace=True)
    #
    # # Average CE of that student/author
    # authors_dict = {}
    # for i in range(len(train_x)):
    #     aid = str(train_x['author_id'][i])
    #     ce = int(str(train_y[i]))
    #     try:
    #         authors_dict[aid].append(ce)
    #     except KeyError:
    #         authors_dict[aid] = [ce]
    #
    # # Dict of most common CE for each author
    # for k in authors_dict:
    #     authors_dict[k] = mean_ce(authors_dict[k])
    #
    # avg_ce_train = []
    # avg_ce_test = []
    # for i in range(len(train_x)):
    #     try:
    #         avg_ce_train.append(authors_dict[str(train_x['author_id'][i])])
    #     except KeyError:
    #         avg_ce_train.append(most_freq_label)
    # for i in range(len(test_x)):
    #     try:
    #         avg_ce_test.append(authors_dict[str(test_x['author_id'][i])])
    #     except KeyError:
    #         avg_ce_test.append(most_freq_label)

    # features_train = pd.DataFrame()
    # features_test = pd.DataFrame()
    #
    # # Add to feature space
    # features_train['avg_ce'] = avg_ce_train
    # features_test['avg_ce'] = avg_ce_test
    #
    # features_train['is_reply'] = train_x['is_reply']
    # features_test['is_reply'] = test_x['is_reply']
    #
    # features_train['jaccard_sims'] = train_x['jaccard_sims']
    # features_test['jaccard_sims'] = test_x['jaccard_sims']
    #
    # features_train['length'] = train_x['length']
    # features_test['length'] = test_x['length']
    #
    # features_train['Confusion_label'] = train_x['Confusion_label']
    # features_test['Confusion_label'] = test_x['Confusion_label']
    #
    # # Tags not including: help, confused and frustrated
    # features_train['#useful'] = train_x['#useful']
    # features_test['#useful'] = test_x['#useful']
    # features_train['#idea'] = train_x['#idea']
    # features_test['#idea'] = test_x['#idea']
    # features_train['#question'] = train_x['#question']
    # features_test['#question'] = test_x['#question']
    # features_train['#interested'] = train_x['#interested']
    # features_test['#interested'] = test_x['#interested']
    # features_train['#curious'] = train_x['#curious']
    # features_test['#curious'] = test_x['#curious']
    #
    # run_rf(features_train, train_y, features_test, test_y)

    features = pd.DataFrame()

    features['is_reply'] = data_x['is_reply']
    features['jaccard_sims'] = data_x['jaccard_sims']
    features['length'] = data_x['length']
    features['Confusion_label'] = data_x['Confusion_label']

    # Tags not including: help, confused and frustrated
    features['#useful'] = data_x['#useful']
    features['#idea'] = data_x['#idea']
    features['#question'] = data_x['#question']
    features['#interested'] = data_x['#interested']
    features['#curious'] = data_x['#curious']

    run_rf(features, labels)

    # print(rf_out)


if __name__ == '__main__':
    main()
