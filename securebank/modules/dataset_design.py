import pandas as pd
from sklearn.model_selection import train_test_split

class Dataset_Designer:
    def extract(self, raw_dataset_filename: str):
        """
        Reads the parquet raw data file.

        :param raw_dataset_filename: Path to the raw dataset parquet file.
        :return: A pandas DataFrame containing the raw dataset.
        """
        raw_dataset = pd.read_parquet(raw_dataset_filename)
        return raw_dataset

    def sample(self, raw_dataset: pd.DataFrame):
        """
        Partitions the data into training dataset, test dataset, etc.

        :param raw_dataset: DataFrame containing the raw dataset.
        :return: A list of pandas DataFrames containing partitioned data.
        """
        # Partitioning the dataset into training (70%), validation (15%), and test (15%) sets
        train_data, temp_data = train_test_split(raw_dataset, test_size=0.3, random_state=42)
        validation_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)
        
        partitioned_data = [train_data, validation_data, test_data]
        return partitioned_data

    def describe(self, *args, **kwargs):
        """
        Computes significant quality metrics of the transformed dataset.

        :param *args, **kwargs: Additional arguments for flexibility.
        :return: A dictionary containing dataset version, storage path, and important dataset description.
        """
        # Placeholder for the actual dataset version and storage path
        version_name = kwargs.get('version', 'v1.0')
        storage_path = kwargs.get('storage', './securebank/storage/partitioned_data/')

        # Computing basic statistics and quality metrics
        description = {
            'num_rows': args[0].shape[0],
            'num_columns': args[0].shape[1],
            'column_info': args[0].dtypes.to_dict(),
            'missing_values': args[0].isnull().sum().to_dict(),
            'fraud_count': args[0]['is_fraud'].sum(),
        }

        return {
            'version': version_name,
            'storage': storage_path,
            'description': description
        }

    def load(self, partitioned_data: list, output_filename: str):
        """
        Saves the partitioned data into storage in parquet format.

        :param partitioned_data: List of DataFrames containing partitioned data.
        :param output_filename: Base path where parquet files should be saved.
        """
        # Save each partition (train, validation, test) into separate parquet files
        partition_names = ['train', 'validation', 'test']
        for data, name in zip(partitioned_data, partition_names):
            data.to_parquet(f"{output_filename}_{name}.parquet", index=False)
