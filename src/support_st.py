import pickle
import pandas as pd
import numpy as np
from src.support_preprocess import Encoding
from sklearn.preprocessing import StandardScaler

def load_models():
    with open('models/forest_model.pkl', 'rb') as f:
        model = pickle.load(f)
    return model


def load_options():
    with open('models/options/municipality.pkl', 'rb') as f:
        municipalities = pickle.load(f)

    with open('models/options/propertyType.pkl', 'rb') as f:
        types = pickle.load(f)

    with open('models/options/provinces.pkl', 'rb') as f:
        provinces = pickle.load(f)

    return municipalities, types, provinces


def ensure_columns(df, required_columns):

    for column in required_columns:
        if column not in df.columns:
            df[column] = 0
    return df[required_columns]


def get_prediction(model, propertyType, size, exterior, rooms, bathrooms, distance, floor, municipality, province, hasLift, numPhotos):
      
    # New predictions
    new_house = pd.DataFrame({
        'propertyType': [propertyType],
        'size': [size],
        'exterior': [exterior],
        'rooms': [rooms],
        'bathrooms': [bathrooms],
        'distance': [distance],
        'floor': [floor],
        'municipality': [municipality],
        'province': [province],
        'hasLift' : [hasLift],
        'numPhotos' : [numPhotos]
    })


    df_new = pd.DataFrame(new_house)
    df = df_new.copy()

    encoding_methods = {"onehot": ['propertyType'],
                    "target": [],
                    "ordinal" : {
                        'floor': ['ss', 'st', 'bj', 'en', '1', '2', '3', '4', '5', '6', '7', '8', '14', 'unknown']
                        },
                    "frequency": ['municipality', 'province', 'hasLift']
                    }

    #df["price"] = np.nan
    encoder = Encoding(df, encoding_methods, None)

    df = encoder.execute_all_encodings() 

    required_columns = [
        'size', 'exterior', 'rooms', 'bathrooms', 'distance',
        'municipality', 'province', 'hasLift', 'numPhotos',
        'propertyType_chalet', 'propertyType_countryHouse',
        'propertyType_duplex', 'propertyType_flat', 'propertyType_penthouse',
        'propertyType_studio', 'floor'
    ]

    df = ensure_columns(df, required_columns)

   # StandardScaler
    numeric_features = ['size', 'rooms', 'distance', 'numPhotos']
    numeric_transformer = StandardScaler()

    scaled_data = numeric_transformer.fit_transform(df[numeric_features])
    df[numeric_features] = scaled_data

    #df.drop(columns="price",inplace=True)

    pred = model.best_model['random_forest'].predict(df)
    return pred
