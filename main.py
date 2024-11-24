import warnings

import pandas as pd
import xgboost as xgb
from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.metrics import classification_report
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore", message=r".*Falling back to prediction using DMatrix.*")
warnings.filterwarnings("ignore")


def load_and_clean_data():
    df = pd.read_csv("data/creditcard_2023.csv")
    df = df.drop(columns=["id"], axis=1)
    df = df.dropna()

    features = [col for col in df.columns if col != 'Class']
    df = df.drop_duplicates(subset=features)

    return df


def split_features_target(df):
    X = df.drop(columns=['Class'], axis=1)
    y = df['Class']
    return X, y


def preprocess_data(X_train, X_test):
    imputer = SimpleImputer()
    X_train = imputer.fit_transform(X_train)
    X_test = imputer.transform(X_test)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    pca = PCA(n_components=22)
    X_train = pca.fit_transform(X_train)
    X_test = pca.transform(X_test)

    return X_train, X_test


def prepare_data():
    df = load_and_clean_data()
    X, y = split_features_target(df)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        stratify=y,
    )

    X_train, X_test = preprocess_data(X_train, X_test)

    return X_train, X_test, y_train, y_test


def grid_search(model, X_train, y_train):
    param_grid = {
        'max_depth': [5, 8],
        'learning_rate': [0.1],
        'n_estimators': [50, 100, 120],
        'subsample': [0.6, 0.8],
        'colsample_bytree': [0.8],
    }

    scoring = 'accuracy'

    gs = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring=scoring,
        cv=5,
        verbose=4,
        n_jobs=-1
    )

    print("Starting grid search...")
    gs.fit(X_train, y_train)

    print(f"Best CV Score (Accuracy): {gs.best_score_}")
    print(f"Best Hyperparameters: {gs.best_params_}")

    best_model = gs.best_estimator_

    return best_model, gs.best_params_


def test(model, X_test, y_test):
    y_pred = model.predict(X_test)
    y_pred = (y_pred >= 0.5).astype(int)

    print(classification_report(y_test, y_pred))

    from common import plot_confusion_matrix
    plot_confusion_matrix(y_test, y_pred)


def final_train(params, X_train, y_train):
    X_train, X_val, y_train, y_val = train_test_split(
        X_train, y_train,
        test_size=0.2,
        stratify=y_train,
    )

    print("Starting training...")

    final_model = xgb.XGBClassifier(
        **params,
        objective='binary:logistic',
        eval_metric="logloss",
        enable_categorical=True
    )

    final_model.fit(
        X_train, y_train,
        eval_set=[(X_train, y_train), (X_val, y_val)],
        verbose=4
    )

    print("Final training complete.")

    return final_model


def train():
    X_train, X_test, y_train, y_test = prepare_data()

    model = xgb.XGBClassifier(
        objective='binary:logistic',
        eval_metric='logloss',
        enable_categorical=True
    )

    model, params = grid_search(model, X_train, y_train)
    model = final_train(params, X_train, y_train)
    test(model, X_test, y_test)

    model.save_model('models/best_model.model')
    print("XGBoost model saved successfully.")


def main():
    train()


if __name__ == '__main__':
    main()
