from create_features import make_feature_df
from datetime import datetime
from utils.load_config import load_config

# Load fs config
config_path = "/Users/phil/philclarkphd/sleep/data_processing/fs_config.json"
config = load_config(config_path)


training_data_path = config["paths"]["training_data_path"]
feature_store_path = config["paths"]["feature_store_path"]

current_time = datetime.strftime(datetime.today(), "%Y-%m-%d_%H-%M-%S")
feature_table_name = f"features_{current_time}.csv"
save_path = feature_store_path + "/" + feature_table_name

# Call functions to make feature_df
df_features = make_feature_df(training_data_path)
print(df_features.shape)
print(df_features.head())
print(feature_table_name)

# Save feature table in feature store
df_features.to_csv(save_path)
