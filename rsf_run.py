from RSF import RandomSurvivalForest
import pandas as pd


def future_prediction(year, prev_entity, df, rd, train_year):
    if year == train_year:
        df_testing_entity = df[df['observation_year'] == year]        # only data to obtain from DF
    else:
        df_testing_entity = prev_entity                                 # Use previously obtained data
        df_testing_entity.loc[:,'age'] = df_testing_entity['age'] + 1         # Migrate to next year

    upper, lower, prediction = rd.predict_proba(df_testing_entity[['n_failures', 'laid_year', 'size', 'age']])
    prediction_d = pd.DataFrame(prediction, index=df_testing_entity.index)
    prediction_d.loc[:,0]=1.0-prediction_d.loc[:,0]
    df_testing_entity.loc[:,'n_failures'] = df_testing_entity.loc[:,'n_failures'] + prediction_d.loc[:,0]

    return df_testing_entity, upper, lower, prediction


def train_predict_write(file_name, out_folder, train_year):
    df = pd.read_csv(file_name + '.csv')
    df_training_entity = df[df['observation_year'] <= train_year]  # until 2010
    rd = RandomSurvivalForest(n_trees=100)
    estimators = rd.fit(df_training_entity[['n_failures', 'laid_year', 'size']], df_training_entity[['age', 'Failed']])
    prev_test_entity = []
    for x in range(train_year, 2018):
        prev_test_entity, up_c, low_c, prediction = future_prediction(x, prev_test_entity, df, rd, train_year)
        df_n = pd.DataFrame()
        df_n.loc[:,'y'] = pd.Series(prediction, index=prev_test_entity.index)
        df_n = pd.concat([prev_test_entity,df_n], axis=1)
        df_n.to_csv(out_folder+'rsf'+str(train_year)+'_'+str(x)+'.csv', index=False)


train_predict_write('data/train1', 'Output/train1/', 2011)
train_predict_write('data/train2', 'Output/train2/', 2013)
train_predict_write('data/train3', 'Output/train3/', 2011)
