import pickle
import pandas as pd
import numpy as np
import ast

from sklearn.feature_extraction import DictVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold
from sklearn.metrics import root_mean_squared_error

alpha=0.1

def read_data():
    ## Data Reading
    return pd.read_csv("data/funda-rotterdam-11-18-2024.csv")

def standardize_column_names(df_raw):
    ## Standardize Column Names
    df_raw.columns = (
        df_raw.columns
        .str.replace(".", "_")
        .str.replace("blikvanger", "eyecatcher")
    )

## Data Preparation and Cleaning
def list_to_ohe(df, column):
    '''Converts a column with list-like string values into one-hot encoded columns.'''
    try:
        # Parse string as list and create one-hot encoded columns
        return (
            df
            .assign(**{f"{column}_{v}": df[column].apply(ast.literal_eval).apply(lambda x: v in x) for v in set(df[column].apply(ast.literal_eval).sum())})
            .drop(columns=column)  # Drop the original column after encoding
        )
    except ValueError:
        # Fallback if values are not in stringified list format
        return (
            df
            .assign(**{f"{column}_{v}": df[column].apply(lambda x: v in x) for v in set(df[column].sum())})
            .drop(columns=column)
        )

def list_to_ohe_multiple(df, columns):
    '''Applies one-hot encoding to multiple columns with list-like values.'''
    for column in columns:
        df = list_to_ohe(df, column)  # Update DataFrame with each column's one-hot encoding
    return df

def log1p_transform(df, columns):
    '''Applies a log(1+x) transformation to specified numeric columns.'''
    if not columns:
        return df  # Return original DataFrame if no columns specified
    return df.assign(**{c: np.log1p(df[c]) for c in columns})  # Apply log1p transformation

def prepare_df(df_raw, log1p_columns=None):
    '''Prepares the raw DataFrame for sklearn.'''
    return (
        df_raw
        # Clean columns and transform to the right data types
        .assign(
            object_type_specifications_appartment_type=df_raw.object_type_specifications_appartment_type.fillna("not_applicable").astype("string"),
            object_type_specifications_house_type=df_raw.object_type_specifications_house_type.fillna("not_applicable").astype("string"),
            object_type_specifications_house_orientation=df_raw.object_type_specifications_house_type.fillna("not_applicable").astype("string"),
            exterior_space_garden_size=pd.to_numeric(df_raw.exterior_space_garden_size.fillna(0), downcast="integer"),
            publish_date=pd.to_datetime(df_raw.publish_date, format="mixed"),
            floor_area=pd.to_numeric(df_raw.floor_area.str.extract(r"\[(\d+)\]")[0], downcast="integer"),
            plot_area=pd.to_numeric(df_raw.plot_area.str.extract(r"\[(\d+)\]")[0], downcast="integer"),
            garage_total_capacity=df_raw.garage_total_capacity.fillna(0).astype("string"),
            price_selling_price=pd.to_numeric(df_raw.price_selling_price.str.extract(r"\[(\d+)\]")[0], downcast="integer"),
        )
        # filter valid prices
        .query('''price_selling_price > 1''')
        # Apply log1p transformation to numeric columns specified in log1p_columns
        .pipe(log1p_transform, columns=log1p_columns)
        # Apply one-hot encoding to list-like columns
        .pipe(list_to_ohe_multiple, columns=["exterior_space_type", "exterior_space_garden_orientation", "surrounding", "garage_type", "amenities", "accessibility"])
        # Set final data types for specific columns
        .astype({
            "number_of_rooms": "string",
            "number_of_bedrooms": "string",
            "address_municipality": "string",
            "price_selling_price_condition": "string",
            "construction_type": "string",
            "construction_period": "string",
            "object_type": "string",
            "energy_label": "string",
        })
        # Drop unused columns
        .drop(
            columns=[
                "publish_date",
                "address_neighbourhood", 
                "address_wijk", 
                "address_province", 
                "address_street_name",
                "eyecatcher_text",
                "project_url",
                "offering_type",
            ]
        )
    )

def prepare_split(df):
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=1)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=1)
    df_full_train = df_full_train.reset_index(drop=True)

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    y_train = df_train.price_selling_price.values
    y_val = df_val.price_selling_price.values
    y_test = df_test.price_selling_price.values

    del df_train["price_selling_price"]
    del df_val["price_selling_price"]
    del df_test["price_selling_price"]

    return df_full_train, df_test, df_train, df_val, y_train, y_val, y_test

# Model Training

## Linear Regression
def train(df, y, model, scaling):
    dicts = df.to_dict(orient="records")
    
    dv = DictVectorizer(sparse=False)
    X_train = dv.fit_transform(dicts)

    if scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
    
    model.fit(X_train, y)
    
    return dv, model

def predict(df, dv, model):
    dicts = df.to_dict(orient="records")
    X = dv.transform(dicts)
    y_pred = model.predict(X)
    return y_pred

def cross_validate(df_full_train, model, n_splits, scaling=False):

    kfold = KFold(n_splits=n_splits, shuffle=True, random_state=1)

    train_scores = []
    val_scores = []

    for train_idx, val_idx in kfold.split(df_full_train):
        y_train = df_full_train.price_selling_price.iloc[train_idx].values
        y_val = df_full_train.price_selling_price.iloc[val_idx].values

        df_train = df_full_train.drop(columns="price_selling_price").iloc[train_idx]
        df_val = df_full_train.drop(columns="price_selling_price").iloc[val_idx]

        dv, model_after_fitting = train(df_train, y_train, model, scaling=scaling)
        y_train_pred = predict(df_train, dv, model_after_fitting)
        y_val_pred = predict(df_val, dv, model_after_fitting)

        train_rmse = root_mean_squared_error(y_train, y_train_pred)
        val_rmse = root_mean_squared_error(y_val, y_val_pred)

        train_scores.append(train_rmse)
        val_scores.append(val_rmse)

    print(f"Train RMSE (mean and std): {np.mean(train_scores).round(0)} and {np.std(train_scores).round(0)}")
    print(f"Validation RMSE (mean and std): {np.mean(val_scores).round(0)} and {np.std(val_scores).round(0)}")

def final_training(df_full_train, alpha):
    # Final Model Fitting on full_train (train+validation) and prediction on test
    model = Ridge(alpha=alpha)
    dv, model_full_train = train(df_full_train.drop(columns="price_selling_price"), df_full_train.price_selling_price.values, model, scaling=False)

    y_pred_full_train = predict(df_full_train, dv, model_full_train)
    y_pred_test = predict(df_test, dv, model_full_train)
    print(f"RMSE full train: {root_mean_squared_error(df_full_train.price_selling_price.values, y_pred_full_train)}")
    print(f"RMSE test: {root_mean_squared_error(y_test, y_pred_test)}")
    return dv, model_full_train

def save_model(dv, model):
    output_file = f"ridge_model_alpha={alpha}.bin"
    with open(output_file, "wb") as f_out:
        '''Saving the model'''
        pickle.dump((dv, model), f_out)

if __name__=="__main__":
    print("Data Reading, Cleaning, and Preparation")
    df_raw = read_data()
    standardize_column_names(df_raw)
    df = prepare_df(df_raw)
    df_full_train, df_test, df_train, df_val, y_train, y_val, y_test = prepare_split(df)
    model = Ridge(alpha=alpha)
    print("Cross-Validation")
    cross_validate(df_full_train, model, 5)
    dv, model_full_train = final_training(df_full_train, alpha=0.1)
    print("Saving Model")
    save_model(dv, model_full_train)
