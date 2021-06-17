import pandas as pd
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
########################################################################################################################


def feat_imp(f_name):

    data = pd.read_csv(f_name + '.csv')
    data.columns = ['Failed', 'n_failures', 'age', 'laid_year', 'size', 'pipe_type', 'length', 'observation_year']
    index_names = data[data.age < 0].index
    data.drop(index_names, inplace=True)
    data.pipe_type = pd.Categorical(data.pipe_type).codes

    x = data.iloc[:, 1:]
    y = data.iloc[:, 0]
    # print(x)
    # print(y)
    best_features = SelectKBest(score_func=chi2, k=6)
    fit = best_features.fit(x, y)
    df_scores = pd.DataFrame(fit.scores_)
    df_columns = pd.DataFrame(x.columns)
    feature_scores = pd.concat([df_columns, df_scores], axis=1)
    feature_scores.columns = ['Specs', 'Score']
    print('**************', f_name, '****************')
    print(feature_scores.nlargest(6, 'Score'))
    print('**************************************')
########################################################################################################################


feat_imp('data/train1')
feat_imp('data/train2')
feat_imp('data/train3')
