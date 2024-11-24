# Data processing  
# -----------------------------------------------------------------------
import pandas as pd
import numpy as np

# Ignore warnings  
# -----------------------------------------------------------------------
import warnings  
warnings.filterwarnings("ignore") 

# Machine learning imports
# -----------------------------------------------------------------------
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder
from category_encoders import TargetEncoder


class Encoding:
    """
    A class to perform various encoding techniques on a DataFrame.

    Attributes:
    - df (pd.DataFrame): The input DataFrame containing the data to be encoded.
    - encoding_methods (dict): A dictionary specifying the encoding methods and their configurations for different columns.
    - target_variable (str): The name of the target variable used for target encoding.

    Methods:
    - __init__(df: pd.DataFrame, encoding_methods: dict, target_variable: str): Initializes the Encoding object with the DataFrame, encoding methods, and target variable.
    - one_hot_encoding(): Applies one-hot encoding to specified columns in the DataFrame.
    - target_encoding(): Applies target encoding to specified columns in the DataFrame.
    - ordinal_encoding(): Applies ordinal encoding to specified columns in the DataFrame based on defined category orders.
    - frequency_encoding(): Applies frequency encoding to specified columns in the DataFrame.
    - execute_all_encodings(): Executes all encoding methods sequentially, skipping undefined ones and handling exceptions to ensure subsequent methods are executed.
    """

    def __init__(self, df: pd.DataFrame, encoding_methods: dict, target_variable: str):
        """
        Initializes the object with a DataFrame, encoding methods, and target variable.

        Parameters:
        - df (pd.DataFrame): The input DataFrame containing the data.
        - encoding_methods (dict): A dictionary specifying the encoding methods for different columns.
        - target_variable (str): The name of the target variable in the DataFrame.
        """
        self.df = df
        self.encoding_methods = encoding_methods
        self.target_variable = target_variable
    
    
    def one_hot_encoding(self):
        """
        Applies one-hot encoding to specified columns in the DataFrame.

        Parameters:
        - None

        Returns:
        - (pd.DataFrame): The updated DataFrame with one-hot encoded columns added and the original columns removed.
        """
        # Get columns to one hot encode
        cols = self.encoding_methods.get("onehot", [])

        if cols:
            one_hot_encoder = OneHotEncoder()

            # Perform one-hot encoding
            encoded_data = one_hot_encoder.fit_transform(self.df[cols])

            # Create a DataFrame for the encoded columns
            df_encoded = pd.DataFrame(encoded_data.toarray(), columns=one_hot_encoder.get_feature_names_out())

            # Concatenate the encoded columns to the original DataFrame
            self.df = pd.concat([self.df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

            # Drop the original columns that were one-hot encoded
            self.df.drop(columns=cols, inplace=True)

        return self.df

    
    def target_encoding(self):
        """
        Applies target encoding to specified columns in the DataFrame.

        Parameters:
        - None

        Returns:
        - (pd.DataFrame): The updated DataFrame with target-encoded values replacing the original columns.
        """
        # Get columns to target encode
        cols = self.encoding_methods.get("target", [])

        if cols:
            # Validate target variable presence
            if self.target_variable not in self.df.columns:
                raise ValueError(f"Target variable '{self.target_variable}' is not in the DataFrame.")

            # Initialize the TargetEncoder
            target_encoder = TargetEncoder()

            # Perform target encoding for the specified columns
            encoded_data = target_encoder.fit_transform(self.df[cols], self.df[self.target_variable])
            
            # Replace the original columns with the encoded values
            self.df[cols] = encoded_data

        return self.df

    
    def ordinal_encoding(self):
        """
        Applies ordinal encoding to specified columns in the DataFrame based on defined category orders.

        Parameters:
        - None

        Returns:
        - (pd.DataFrame): The updated DataFrame with ordinal-encoded values replacing the original columns.
        """
        # Retrieve columns and their respective category orders
        cols = self.encoding_methods.get("ordinal", {})

        if cols:
            # Ensure all specified columns exist in the DataFrame
            missing_cols = [col for col in cols.keys() if col not in self.df.columns]
            if missing_cols:
                raise ValueError(f"The following columns are missing from the DataFrame: {missing_cols}")

            # Extract category orders from the dictionary
            categories_order = list(cols.values())

            # Initialize the OrdinalEncoder with specified options
            ordinal_encoder = OrdinalEncoder(
                categories=categories_order,
                dtype=float,
                handle_unknown="use_encoded_value",
                unknown_value=np.nan
            )

            # Perform the ordinal encoding
            encoded_data = ordinal_encoder.fit_transform(self.df[list(cols.keys())])

            # Drop the original columns that were ordinal encoded
            self.df.drop(cols, axis=1, inplace=True)

            # Create a DataFrame for the encoded columns
            df_encoded = pd.DataFrame(encoded_data, columns=ordinal_encoder.get_feature_names_out())

            # Concatenate the encoded columns to the original DataFrame
            self.df = pd.concat([self.df.reset_index(drop=True), df_encoded.reset_index(drop=True)], axis=1)

        return self.df


    def frequency_encoding(self):
        """
        Applies frequency encoding to specified columns in the DataFrame.

        Parameters:
        - None

        Returns:
        - (pd.DataFrame): The updated DataFrame with frequency-encoded values for the specified columns.
        """
        # Get columns to frequency encode
        cols = self.encoding_methods.get("frequency", [])

        if cols:
            # Apply encoding for every categroy
            for category in cols:
                # Compute the frequency of the categories
                frecuency = self.df[category].value_counts(normalize=True)
                # Map values
                self.df[category] = self.df[category].map(frecuency)
        
        return self.df
    

    def execute_all_encodings(self):
        """
        Executes all encoding methods sequentially: One-Hot Encoding, Target Encoding, Ordinal Encoding, and Frequency Encoding.

        The method skips any encoding types not defined in `self.encoding_methods` and handles exceptions during the execution of each encoding type to ensure subsequent encodings are still performed.

        Parameters:
        - None

        Returns:
        - (pd.DataFrame): The updated DataFrame after applying all specified encoding methods.
        """
        # Check and execute One-Hot Encoding
        if "onehot" in self.encoding_methods:
            try:
                self.one_hot_encoding()
            except Exception as e:
                print(f"Error during One-Hot Encoding: {e}")

        # Check and execute Target Encoding
        if "target" in self.encoding_methods:
            try:
                self.target_encoding()
            except Exception as e:
                print(f"Error during Target Encoding: {e}")

        # Check and execute Ordinal Encoding
        if "ordinal" in self.encoding_methods:
            try:
                self.ordinal_encoding()
            except Exception as e:
                print(f"Error during Ordinal Encoding: {e}")

        # Check and execute Frequency Encoding
        if "frequency" in self.encoding_methods:
            try:
                self.frequency_encoding()
            except Exception as e:
                print(f"Error during Frequency Encoding: {e}")

        # Return the updated DataFrame after all encodings
        return self.df
