import os
# Set the environment variables directly in the script (for testing)
os.environ['KAGGLE_USERNAME'] = 'samiullaheng12'
os.environ['KAGGLE_KEY'] = 'cb625748b4196d2af7cefee87025113b'

# Initialize Kaggle API
from kaggle.api.kaggle_api_extended import KaggleApi
import traceback
api = KaggleApi()
try:
    api.authenticate()
    print("Kaggle API authentication successful!")
except Exception as e:
    print("Failed to authenticate with Kaggle API.")
    print(e)

# Define the dataset and destination path
dataset_name = 'paultimothymooney/chest-xray-pneumonia'  # Dataset path from Kaggle URL
destination_folder = 'data/raw'  # Replace with the desired folder path

# Ensure the destination folder exists
if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

# Download the dataset
try:
    print(f"Downloading dataset {dataset_name} to {destination_folder}")
    api.dataset_download_files(dataset_name, path=destination_folder, unzip=True)
    print(f"Dataset downloaded and extracted to {destination_folder}")
except Exception as e:
    print("Error downloading dataset:")
    print(e)
    traceback.print_exc()