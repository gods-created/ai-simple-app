from os.path import exists
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from joblib import dump, load
import pandas as pd


global MODEL_DIR
MODEL_DIR = 'model.joblib'


def if_file_exists(filedir: str = '') -> bool:
    return exists(filedir)


def prepared_data(filedir: str = '', task: str = 'training') -> list:
    if not filedir.endswith('.csv'):
        return []
    
    table = pd.read_csv(filedir)
    waiting_length = 14 if task == 'training' else 13
    if (columns := len(table.columns)) != waiting_length:
        return []
    
    data = table.iloc[:].fillna(0).values.tolist()

    if task == 'training':
        return [
            [values[:-1] for values in data],
            [values[-1] for values in data]
        ]

    elif task == 'predict':
        return [values[:] for values in data]
    

def update_table(filedir: str = '', predictions: list = []) -> None:
    table = pd.read_csv(filedir)
    table['MEDV'] = predictions
    table.to_csv(filedir, index=False)
    return None
    

def predict(filedir: str = '') -> str:
    if not if_file_exists(MODEL_DIR):
        return 'The model doesn\'t exist'

    incoming_data = prepared_data(filedir, 'predict')
    if not incoming_data:
        return 'Invalid file content'
    
    model = load(MODEL_DIR)
    predictions = model.predict(incoming_data)
    update_table(filedir, predictions)

    return 'The predictions wrote to file'


def training(filedir: str = '') -> str:
    prepared_data_response = prepared_data(filedir)
    if not prepared_data_response:
        return 'Invalid file content'
    
    incoming_data, answers = prepared_data_response
    X_train, _, y_train, _ = train_test_split(incoming_data, answers)
    model = RandomForestRegressor()
    model.fit(X_train, y_train)
    dump(model, MODEL_DIR)

    return 'The model training and saving finished success'