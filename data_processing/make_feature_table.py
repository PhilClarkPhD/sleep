import create_features as cf
from datetime import datetime

BASE_PATH = "/Users/phil/philclarkphd/sleep/sleep_data/training_data/sophie/subset_1"
FEATURE_STORE_DIR = "/Users/phil/philclarkphd/sleep/sleep_data/feature_store"

current_time = datetime.strftime(datetime.today(), "%Y-%m-%d_%H-%M-%S")
FILENAME = f"features_{current_time}.csv"
SAVE_PATH = FEATURE_STORE_DIR + "/" + FILENAME

# Call functions to make feature_df
df_features = cf.make_feature_df(BASE_PATH)
print(df_features.head())

# Save feature table in feature store
df_features.to_csv(SAVE_PATH)
