import pandas as pd
from models import pipeline_regression as regress
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

model_pipelines = {}


def train(df_training_entity, model2save, model_name):
    y = df_training_entity['Failed']
    x = df_training_entity[['n_failures', 'size', 'pipe_type', 'age']]
    md_reg = regress(model_pipelines[model_name])
    print('Start training ...')
    md_reg.train(x, y)
    print('Training for ', model_name, 'is completed.')
    md_reg.save_model(model2save)
    print('Trained model is saved.')


def predict(df_testing_entity, model2read, model_name, path):
    x = df_testing_entity[['n_failures', 'size', 'pipe_type', 'age']]
    md_reg = regress(model_pipelines[model_name])
    md_reg.load_model(model2read)
    y_pr = md_reg.predict(x)
    df_testing_entity['y'] = y_pr
    df_testing_entity.to_csv(path, index=False)


def train_predict_write(file_name, out_folder, train_year):
    df = pd.read_csv(file_name + '.csv')
    df_train = df[df['observation_year'] <= train_year] # up to train year

    # Random Forest
    model_pipelines['RandomForest'] = RandomForestRegressor(n_estimators=100, min_samples_leaf=5, random_state=None,
                                                            max_features=4, n_jobs=-1)
    train(df_train, (file_name), 'RandomForest')
    for x in range((train_year+1), 2018):
        predict(df[df['observation_year'] == x],  (file_name), 'RandomForest',
                (out_folder + 'rf' + str(train_year) + '_' + str(x) + '.csv'))

    # Gradient Boosting
    model_pipelines['GradientBoost'] = GradientBoostingRegressor(n_estimators=100, min_samples_leaf=5,
                                                                 random_state=None, max_features=4)
    train(df_train, (file_name), 'GradientBoost')
    for x in range((train_year+1), 2018):
        predict(df[df['observation_year'] == x], (file_name), 'GradientBoost',
                (out_folder + 'gb' + str(train_year) + '_' + str(x) + '.csv'))


train_predict_write('data/train1', 'Output/train1/', 2011)
train_predict_write('data/train2', 'Output/train2/', 2013)
train_predict_write('data/train3', 'Output/train3/', 2011)
