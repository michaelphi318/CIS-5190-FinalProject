# File: preprocess.py

import pandas as pd
import os
from typing import List, Tuple, Any

def prepare_data(csv_path: str) -> Tuple[List[str], Any]:
    """
    Reads the metadata.csv and returns a list of full image paths
    to be processed by the model.
    """
    # Find the directory
    data_dir = os.path.dirname(csv_path)
    
    # Read the metadata file
    df = pd.read_csv(csv_path)
    
    # Handle the 'file_name' or 'filename' column
    if 'file_name' not in df.columns and 'filename' in df.columns:
        df = df.rename(columns={'filename': 'file_name'})

    # Create a list of full, absolute paths to the images
    X = [os.path.join(data_dir, fname) for fname in df['file_name']]
    
    y = None
    return X, y
