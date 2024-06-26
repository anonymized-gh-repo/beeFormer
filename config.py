from datasets.utils import *
import os
os.environ["KERAS_BACKEND"] = "torch"

config = {
    "ml20m": 
    (
        Dataset("MovieLens20M"), 
        {            
            "filename" : "datasets/ml20m/ratings.csv",
            "item_id_name" : "movieId", 
            "user_id_name" : "userId",
            "value_name" : "rating",
            "timestamp_name" : "timestamp",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "set_all_values_to" : 1.,
            "num_test_users": 10000,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_feather("datasets/ml20m/item_text_descriptions.feather")""",
            "items_item_id_name": "movieId",
            "items_preprocess": """f'{row.llama_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,

        }
    ),
    "goodbooks":
    (
        Dataset("Goodbooks-10k"),
        {
            "raw_data" : """pd.read_csv("datasets/goodbooks/ratings.csv")""",
            "user_id_name":"user_id",
            "item_id_name":"book_id",
            "value_name":"rating",
            "min_value_to_keep" : 4.,
            "user_min_support" : 5,
            "item_min_support" : 1,
            "set_all_values_to" : 1.,
            "num_test_users": 2500,
            "random_state": 42,
            "load_previous_splits": False,
            "items_raw_data": """pd.read_json("datasets/goodbooks/item_text_descriptions.json")""",
            "items_item_id_name": "book_id",
            "items_preprocess": """f'{row.llama_description}'""",
            "coldstart_fraction": 0.1,
            "num_coldstart_items": 2000,
        }
    ),
}