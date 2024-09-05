# Data Pipeline Design for Fraud Detection

## Introduction

The design of the data pipeline for fraud detection at SecureBank involves three key modules: `Raw_Data_Handler`, `Dataset_Designer`, and `Feature_Extractor`. Each module is responsible for distinct steps in preparing the data for model training, ensuring a structured approach to handle data extraction, cleaning, partitioning, and feature engineering. This document provides an overview of the design decisions made for each module and the rationale behind these choices, considering the nature of the data and the problem of fraud detection.

## 1. Raw_Data_Handler Module

### Purpose
The `Raw_Data_Handler` module is responsible for:
- Extracting data from various sources (CSV, Parquet, JSON).
- Cleaning and standardizing the data for further use.
- Merging the extracted datasets into a single comprehensive dataset.

### Design Decisions

#### **Data Extraction from Multiple Formats**
- **Decision**: Use separate functions to read CSV, Parquet, and JSON files.
- **Reasoning**: The raw data is provided in multiple formats, each requiring specific methods for reading. This modular approach allows flexibility to handle various formats efficiently.

#### **Data Merging and Standardization**
- **Decision**: Merge data based on common keys (e.g., `cc_num` for customer information and `trans_num` for transaction and fraud data).
- **Reasoning**: Merging on unique identifiers ensures all relevant data for each transaction is consolidated. This comprehensive view is crucial for detecting patterns associated with fraudulent behavior.

#### **Handling Missing Values**
- **Decision**: Use forward fill (`ffill`) method for handling missing values.
- **Reasoning**: The forward fill method assumes that missing values are likely to be similar to the previous ones, which is a reasonable assumption for temporal transaction data where subsequent transactions might have similar values for certain features.

#### **Data Type Conversion**
- **Decision**: Convert date columns to datetime format.
- **Reasoning**: Fraud detection often relies on time-based patterns (e.g., transactions occurring in quick succession). Correctly formatted date and time fields are essential for extracting temporal features.

## 2. Dataset_Designer Module

### Purpose
The `Dataset_Designer` module is responsible for:
- Partitioning the raw dataset into training, validation, and testing datasets.
- Ensuring that the data partitions are representative and suitable for training robust machine learning models.

### Design Decisions

#### **Data Partitioning**
- **Decision**: Split data into 70% training, 15% validation, and 15% testing sets.
- **Reasoning**: Given the need to develop a model that generalizes well, the chosen split balances the need for sufficient training data while retaining a meaningful proportion for validation and testing. This ratio is commonly used in machine learning to optimize model performance.

#### **Random Sampling with Fixed Seed**
- **Decision**: Use random sampling with a fixed seed (`random_state=42`).
- **Reasoning**: Random sampling ensures that the partitions are representative of the overall dataset. The fixed seed allows reproducibility of results, which is important for model evaluation and comparison.

## 3. Feature_Extractor Module

### Purpose
The `Feature_Extractor` module is responsible for:
- Extracting relevant features from the raw data.
- Transforming and encoding features to formats suitable for machine learning models.

### Design Decisions

#### **Numerical and Categorical Feature Separation**
- **Decision**: Identify and separate numerical features (e.g., transaction amount, location coordinates) and categorical features (e.g., merchant, category).
- **Reasoning**: Numerical and categorical features require different preprocessing steps. Numerical features often benefit from scaling, while categorical features require encoding. Separating them ensures appropriate handling during transformation.

#### **Preprocessing Pipelines**
- **Decision**: Use `Pipeline` and `ColumnTransformer` from scikit-learn to handle preprocessing.
- **Reasoning**: Pipelines provide a structured approach to preprocessing, ensuring that all transformations are applied consistently. `ColumnTransformer` allows different preprocessing steps to be applied to different subsets of features, enhancing modularity and maintainability.

#### 

