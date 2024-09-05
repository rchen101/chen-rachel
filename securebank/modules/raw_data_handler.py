import pandas as pd
import os
import json

class Raw_Data_Handler:
    def extract(self, customer_information_filename: str, transaction_filename: str, fraud_information_filename: str):
        """
        Reads the data sources from the provided filenames.

        :param customer_information_filename: Path to the customer information CSV file.
        :param transaction_filename: Path to the transaction information Parquet file.
        :param fraud_information_filename: Path to the fraud information JSON file.
        :return: Three pandas DataFrames containing customer, transaction, and fraud information respectively.
        """
        customer_information = pd.read_csv(customer_information_filename)

        transaction_information = pd.read_parquet(transaction_filename)
        transaction_information.reset_index(inplace=True)
        
        with open(fraud_information_filename, 'r') as f:
            fraud_data = pd.DataFrame(json.load(f))

        fraud_information = pd.DataFrame(list(fraud_data.items()), columns=['trans_num', 'is_fraud'])
        fraud_information['is_fraud'].fillna(0, inplace=True)
        fraud_information['is_fraud'] = fraud_information['is_fraud'].astype(int)
        fraud_information['trans_num'] = fraud_information['trans_num'].astype(str)    
        
        return customer_information, transaction_information, fraud_information

    def transform(self, customer_information: pd.DataFrame, transaction_information: pd.DataFrame, fraud_information: pd.DataFrame):
        """
        Merges, standardizes, and cleans columns and rows from the three data sources.

        :param customer_information: DataFrame containing customer information.
        :param transaction_information: DataFrame containing transaction information.
        :param fraud_information: DataFrame containing fraud information.
        :return: A merged and cleaned DataFrame suitable for machine learning.
        """
        # Merge dataframes on 'cc_num' and 'trans_num'
        merged_data = transaction_information.merge(customer_information, on='cc_num')
        merged_data = merged_data.merge(fraud_information, on='trans_num')
        
        # Sort by transaction date and time
        merged_data.sort_values(by='trans_date_trans_time', inplace=True)
    
        return merged_data

    def describe(self, *args, **kwargs):
        """
        Computes significant quality metrics of the transformed dataset.

        :param *args, **kwargs: Additional arguments for flexibility.
        :return: A dictionary containing dataset version, storage path, and important dataset description.
        """
        # Placeholder for the actual dataset version and storage path
        version_name = kwargs.get('version', 'v1.0')
        storage_path = kwargs.get('storage', './securebank/storage/processed_data.parquet')

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

    def load(self, raw_data: pd.DataFrame, output_filename: str):
        """
        Saves the transformed data into storage in a parquet format.

        :param raw_data: The DataFrame containing cleaned and processed data.
        :param output_filename: The path where the parquet file should be saved.
        """
        raw_data.to_parquet(output_filename, index=False)
