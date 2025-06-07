import pandas as pd
import numpy as np
import re
import torch
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)
file_path= 'News Title.xls'
def data_load(file_path):
    df = pd.read_excel(file_path)
    title_column = 'News Title'
    category_column = 'Category'
    logger.info(f"Sample data from loaded file:\n{df.head(5)}")

    # Kiem tra cac gia tri bi thieu
    missing_titles = df[title_column].isna().sum()
    missing_categories = df[category_column].isna().sum()
    logger.info(f"Missing titles: {missing_titles}, Missing categories: {missing_categories}")

    # Xoa cac hang co gia tri bi thieu
    df = df.dropna(subset=[title_column, category_column])
    logger.info(f"After removing missing values: {len(df)} rows")

    titles = df[title_column]
    labels = df[category_column]

    # Thay the nhan bang so
    dict_labels = {'Entertainment': 0, 'Business': 1, 'Technology': 2, 'Medical': 3}
    processed_labels = []
    for label in labels:
        if label in dict_labels:
            processed_labels.append(dict_labels[label])
        else:
            logger.warning(f"Unknown category: {label}, assigning to default (0)")
            processed_labels.append(0)  # Default to first category

    return titles, np.array(processed_labels), dict_labels
if __name__ == "__main__":
    titles, labels, dict_labels = data_load(file_path)
