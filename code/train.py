import boto3
import argparse
import joblib
import os
import io
import json

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.datasets import fetch_openml
from sklearn.feature_selection import SelectPercentile, chi2
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV, train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import make_column_selector
from imblearn.pipeline import Pipeline
from imblearn.over_sampling import RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from xgboost import XGBClassifier
from imblearn.ensemble import BalancedBaggingClassifier
from sklearn.metrics import recall_score, precision_score, accuracy_score, f1_score, roc_auc_score, classification_report

np.random.seed(42)


def model_fn(model_dir):
    print('Started Model Function')
    model = joblib.load(os.path.join(model_dir, 'model.joblib'))
    print('End Model Function')
    return model

def input_fn(request_body, request_content_type):
    print('Start input function')
    if request_content_type=='text/csv':
        data = pd.read_csv(io.StringIO(request_body))
        return data
    if request_content_type=='application/json':
        data = pd.read_json(request_body, orient='records')
        return data
    if request_content_type=='application/x-parquet':
        data = pd.read_parquet(io.BytesIO(request_body))
        return data
    raise ValueError('Accept header must be application/json or text/csv or application/x-parquet')

        
def predict_fn(input_object, model):
    print('Start predict function')
    if len(input_object.shape) <= 1:
        input_object = input_object.to_frame().T
    columns = ['offer completed', 'offer received', 'offer viewed', 'transaction']
    predictions = pd.DataFrame(model.predict_proba(input_object), index=input_object.index, columns=columns)
    print('End predict function')
    return predictions

def output_fn(predictions, content_type):
    print('Start output function')
    if content_type=='application/json' or content_type=='application/x-parquet' or content_type=='text/csv':
        result = json.dumps(predictions.to_dict())
        print('End Output function')
        return result



def str2bool(v):
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'y', 'true', 'True', 't', 'y', 1):
        return True
    if v.lower() in ('no', 'n', 'false', 'False', 'f', 'n', 0):
        return False
    else:
        raise argpars.ArgumentTypeError('Boolan value expected')

        

def get_model(kwargs):
    numeric_features = make_column_selector(dtype_include=np.number)
    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median", add_indicator=True)), 
            ("scaler", StandardScaler())
        ]
    )

    categorical_features = make_column_selector(dtype_exclude=np.number)
    categorical_transformer = Pipeline(
        steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )
    
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_features),
            ("cat", categorical_transformer, categorical_features),
        ]
    )
    
    ros = RandomOverSampler(random_state=42, sampling_strategy='auto')
    rus = RandomUnderSampler(random_state=42, sampling_strategy='auto')
    bbc = XGBClassifier(**kwargs)
    
    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor), 
            ("oversample", ros),
            ("undersample", rus),        
            ("model", bbc),
        ]
    )
    return model
    
def main(args):
    
    kwargs = dict(max_depth=args.max_depth, 
                  min_child_weight=args.min_child_weight,
                  subsample=args.subsample, 
                  colsample_bytree=args.colsample_bytree, 
                  eta=args.eta
                 )

    
    train_path = os.path.join(args.train, 'train.csv')
    test_path  = os.path.join(args.test, 'test.csv')
    train = pd.read_csv(train_path, index_col=0)
    test  = pd.read_csv(test_path, index_col=0)

    model =  get_model(kwargs)
    event_to_label = dict(zip(['offer completed', 'offer received', 'offer viewed', 'transaction'], range(4)))

    y_train = train.event.map(event_to_label)
    X_train = train.drop(columns='event')
    
    y_test = test.event.map(event_to_label)
    X_test = test.drop(columns='event')    
    
    model.fit(X_train, y_train)
    y_train_pred = model.predict(X_train)
    y_test_pred  = model.predict(X_test)
    
    metric_train = precision_score(y_train, y_train_pred, average=None).mean()
    metric_test  = precision_score(y_test, y_test_pred, average=None).mean()
    print(f'TRAIN PRECISION: {round(metric_train,3)}')
    print(f'TEST  PRECISION: {round(metric_test,3)}')
        
    joblib.dump(model, os.path.join(args.model_dir, 'model.joblib'))
    
if __name__=='__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--model-dir", type=str, default=os.environ['SM_MODEL_DIR'])
    parser.add_argument("--train", type=str, default=os.environ['SM_CHANNEL_TRAIN'])
    parser.add_argument("--test", type=str, default=os.environ['SM_CHANNEL_TEST'])
    parser.add_argument("--sm-hps", type=str, default=os.environ['SM_HPS'])
    
    parser.add_argument("--max-depth", type=int, default=6), # 0-inf
    parser.add_argument("--min-child-weight", type=int, default=1), # 0-inf    
    parser.add_argument("--subsample", type=float, default=1), # 0-inf        
    parser.add_argument("--colsample_bytree", type=float, default=1), # 0-1        
    parser.add_argument("--eta", type=float, default=1), # 0-1
    args, _ = parser.parse_known_args()

    main(args)