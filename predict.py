import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier


def ordinal_encoder(input_val, feats): 
    feat_val = list(1+np.arange(len(feats)))
    feat_key = feats
    feat_dict = dict(zip(feat_key, feat_val))
    value = feat_dict[input_val]
    return value-1


def get_prediction(data,model):
    target=['Slight Injury','Serious Injury','Fatal injury']
    res = target[model.predict(data)]
    return model.predict(data)