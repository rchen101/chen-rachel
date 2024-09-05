import pandas as pd
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

class Feature_Extractor:
    def extract(self, training_dataset_filename: str, validation_dataset_filename: str, testing_dataset_filename: str):
        """
        Reads the data provided from the file paths.

        :param training_dataset_filename: Path to the training dataset parquet file.
        :param testing_dataset_filename: Path to the testing dataset parquet file.
        :return: Two pandas DataFrames containing the training and testing datasets respectively.
        """
        training_dataset = pd.read_parquet(training_dataset_filename)
        validation_dataset = pd.read_parquet(validation_dataset_filename)
        testing_dataset = pd.read_parquet(testing_dataset_filename)
        return training_dataset, validation_dataset, testing_dataset

    def transform(self, training_dataset: pd.DataFrame, validation_dataset: pd.DataFrame, testing_dataset: pd.DataFrame):
        """
        Converts the dataset into features useful for training.

        :param training_dataset: DataFrame containing the training dataset.
        :param testing_dataset: DataFrame containing the testing dataset.
        :return: A list of pandas DataFrames containing the transformed training and testing datasets.
        """
        # Identifying numerical and categorical features
        categorical_features = training_dataset.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_features = training_dataset.drop(columns=categorical_features).columns.tolist()

        # Defining the preprocessing for numerical and categorical features
        numerical_transformer = Pipeline(steps=[
            ('scaler', StandardScaler())
        ])

        categorical_transformer = Pipeline(steps=[
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])

        # Creating a column transformer that applies the appropriate transformations
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numerical_transformer, numerical_features),
                ('cat', categorical_transformer, categorical_features)
            ]
        )

        # Transforming the training and testing datasets
        X_train = training_dataset.drop('is_fraud', axis=1)
        y_train = training_dataset['is_fraud']
        X_val = validation_dataset.drop('is_fraud', axis=1)
        y_val = validation_dataset['is_fraud']
        X_test = testing_dataset.drop('is_fraud', axis=1)
        y_test = testing_dataset['is_fraud']

        X_train_transformed = preprocessor.fit_transform(X_train)
        X_val_transformed = preprocessor.fit_transform(X_val)
        X_test_transformed = preprocessor.transform(X_test)

        # Converting the transformed datasets into DataFrames
        X_train_transformed_df = pd.DataFrame(X_train_transformed, columns=preprocessor.get_feature_names_out())
        X_val_transformed_df = pd.DataFrame(X_val_transformed, columns=preprocessor.get_feature_names_out())
        X_test_transformed_df = pd.DataFrame(X_test_transformed, columns=preprocessor.get_feature_names_out())

        # Returning transformed features and labels
        transformed_data = [X_train_transformed_df, y_train, X_test_transformed_df, y_val, X_test_transformed_df, y_test]
        return transformed_data

    def describe(self, *args, **kwargs):
        """
        Computes significant quality metrics of the transformed dataset.

        :param *args, **kwargs: Additional arguments for flexibility.
        :return: A dictionary containing dataset version, storage path, and important dataset description.
        """
        # Placeholder for the actual dataset version and storage path
        version_name = kwargs.get('version', 'v1.0')
        storage_path = kwargs.get('storage', './securebank/storage/feature_data/')

        # Computing basic statistics and quality metrics
        description = {
            'num_rows': args[0].shape[0],
            'num_columns': args[0].shape[1],
            'column_info': args[0].dtypes.to_dict(),
            'missing_values': args[0].isnull().sum().to_dict(),
        }

        return {
            'version': version_name,
            'storage': storage_path,
            'description': description
        }
