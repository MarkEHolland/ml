"""
pipeline classification model
"""

from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, Iterable
import json
import joblib
import xgboost as xgb
import pandas as pd
import numpy as np
from loguru import logger
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.impute import KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    confusion_matrix
)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler
from sklearn.impute import MissingIndicator

class ClassificationModel:
    """
    Pipeline Classification model to predict a class n.b. sklearn api compatible models
    """

    def __init__(self, training_config) -> None:
        self.training_config = training_config
        self.xgb_configs = training_config.get("xgb_params", {})
        self.num_cols = training_config.get("numeric_features", [])
        self.cat_cols = training_config.get("categorical_features", [])
        self.target = training_config["target"]
        self.transformed_feature_names = []

        numeric_transformer = Pipeline(
            steps=[
                ('num_imp', SimpleImputer(strategy='constant',fill_value=-1,missing_values=np.nan, add_indicator=False)),
                # ('num_imp', IterativeImputer(estimator=RandomForestRegressor(), initial_strategy='mean',
                #               max_iter=10, random_state=0)), # random forest regressor to impute
                # ('num_imp', KNNImputer(n_neighbors=5, weights="uniform")), # nearest neighbours to impute
                # ('num_rob', RobustScaler()), # n.b. not required for xgboost & if you don't use it then results are easier to interpret
            ]
        )
        
        categorical_transformer = Pipeline(
            steps=[
                ('cat_imp', SimpleImputer(strategy='constant', fill_value='missing', add_indicator=False)), # don't add missing columns
                # ('cat_imp',IterativeImputer(estimator=RandomForestClassifier(), initial_strategy='most_frequent',
                #               max_iter=10, random_state=0)), # random forest classifier to impute
                ('cat_ohe', OneHotEncoder(handle_unknown='ignore', sparse=False)) # don't set up as sparse matrix
            ]
        )
        
        self.ct = ColumnTransformer(
            transformers=[
                ("pp_num", numeric_transformer, self.num_cols),
                ("pp_cat", categorical_transformer, self.cat_cols),
            ],
            remainder="drop",
        )

        self.cv_results = None

        self.pipe_final = Pipeline(
            steps=[
                ("preprocessor", self.ct),
                ("classifier", xgb.XGBClassifier(**self.xgb_configs)),
            ]
        )

        
    def get_transformed_feature_names(self):
        """Generate tranformed feature names.
        
        Args:
        
        Returns:
            transformed_feature_names []: List of tranformed column names.
        """
        
        transformed_cat_feature_names = self.pipe_final.named_steps['preprocessor'].named_transformers_['pp_cat'].named_steps['cat_ohe'].get_feature_names_out(self.cat_cols)
        self.transformed_feature_names = self.num_cols + list(transformed_cat_feature_names)
        
        return self.transformed_feature_names
 

    def transform(self, input_data: pd.DataFrame):
        """Transform given training data.
        
        Args:
            training_data(pd.DataFrame):
        """
        tranformed_input_data = self.pipe_final["preprocessor"].transform(input_data)
        
        return(tranformed_input_data)
        
    
    def fit(self, input_data: pd.DataFrame):
        """Train the models given some training data.

        Args:
            training_data (pd.DataFrame): Training data.
        """

        logger.info("Starting training")
        X = input_data[self.num_cols + self.cat_cols]
        y = input_data[self.target]

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=40)

        self.cv_results = cross_validate(
            self.pipe_final,
            X,
            y,
            cv=cv,
            scoring=(
                "f1",
                "balanced_accuracy",
                "accuracy",
                "recall",
                "precision",
                "roc_auc",
            ),
        )
        self.pipe_final.fit(X, y)
        logger.info("Training complete for all models.")
        # print(f"""\nxgb parameters:-\n {self.pipe_final.named_steps['classifier'].get_xgb_params()}""")

    def predict(self, input_data: pd.DataFrame) -> pd.DataFrame:
        """Predict using all the models given some dates.

        Args:
            input_data (pd.DataFrame): Input data.

        Returns:
            pd.DataFrame: DataFrame of predictions.
        """

        prediction_data = input_data[self.num_cols + self.cat_cols]
        predictions_pr = self.pipe_final.predict_proba(prediction_data)
        predictions = self.pipe_final.predict(prediction_data)

        logger.info(f"Computed predictions for {len(input_data)} records")
        predictions_df = input_data.copy()
        predictions_df["prediction_probability"] = predictions_pr[:, 1]
        predictions_df["prediction"] = predictions

        predictions_df["prediction_datetime"] = datetime.now()
        return predictions_df

        ### methods for visualisation
# #    def fit(self, X: pd.DataFrame, y: pd.Series, **kwargs: dict) -> None:
# #        self.model.fit(X, y, **kwargs)
    
#     def visual_predict(self, input_data: pd.DataFrame) -> Iterable:
#         prediction_data = input_data[self.num_cols + self.cat_cols]
#         return self.model.predict(prediction_data)

#     def visual_predict_proba(self, input_data: pd.DataFrame) -> Iterable:
#         prediction_data = input_data[self.num_cols + self.cat_cols]
#         return self.model.predict_proba(prediction_data)    
# #     ### end methods for visualisation
    
    
    def evaluate(self, input_data: pd.DataFrame) -> Tuple[pd.DataFrame, dict]:
        """Predict using all the models on the dates in the test data,
        compute metrics for the groundtruth data and the predictions.

        Args:
            input_data (pd.DataFrame): Input data.

        Returns:
            Tuple[pd.DataFrame, pd.DataFrame]:
                - DataFrame of predictions.
                - dict of metrics.
        """
        y_true = input_data[self.target]
        X = input_data[self.num_cols + self.cat_cols]
        predictions_prob = self.pipe_final.predict_proba(X)
        predictions = self.pipe_final.predict(X)
        metrics = dict(
            f1_score=f1_score(y_true=y_true, y_pred=predictions),
            roc_auc=roc_auc_score(y_true=y_true, y_score=predictions_prob[:, 1]),
            precision=precision_score(y_true=y_true, y_pred=predictions),
            recall=recall_score(y_true=y_true, y_pred=predictions),
            balanced_acc=balanced_accuracy_score(y_true=y_true, y_pred=predictions),
            accuracy=accuracy_score(y_true=y_true, y_pred=predictions)
        )

        logger.info("Computed metrics...")
        return predictions_prob, metrics

    def load_model(self, file_name: str) -> None:
        """Load model from a source.

        Args:
            file_name (str): Path were model will be loaded.
        """
        self.pipe_final = joblib.load(file_name)

        logger.info(f"Loaded sklearn pipeline from {file_name}.")

    def save_model(self, file_name: str):
        """Save model to a target path.

        Args:
            file_name (str): Path where model will be saved.
        """
        directory = Path(file_name).parent.absolute()
        directory.mkdir(parents=True, exist_ok=True)

        # save model
        joblib.dump(self.pipe_final, file_name, compress=3)
        logger.info(f"Saved model to {file_name}.")
        
        # save model parameters
        with open(file_name+'.json', 'w') as fp:
            json.dump(self.training_config, fp)
        logger.info(f"Saved training parameters to {file_name+'.json'}.")

    @property
    def _feature_importance(self, feature_df):
        """Return the names and importances of the features used by the classifier.
        Property decorator allows access to the model instance
        """
        # feature_importances = self.pipe_final.named_steps['classifier'].feature_importances_
        feature_importances = self.named_steps['classifier'].feature_importances_
        # categorical_feature_names = self.pipe_final.named_steps['preprocessor'].named_transformers_['pp_cat'].named_steps['cat_ohe'].get_feature_names(input_features=self.cat_cols)
        categorical_feature_names = self.named_steps['preprocessor'].named_transformers_['pp_cat'].named_steps['cat_ohe'].get_feature_names(input_features=self.cat_cols)        
        return feature_importances, categorical_feature_names
    
    @property
    def model_instance(self):
        """Return the model instance
        """
        return self.pipe_final

