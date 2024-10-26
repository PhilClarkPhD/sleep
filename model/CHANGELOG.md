# 2024-10-26
## TODO
- Issue: features are generated at run-time in Mora, w/ hardcoded feature selection. Leads to incompatibility between model artifacts and Mora features.
- Solution: Change data_processing into feature store and track it w/ versioning. In Mora, reference relevant feature store version for feature selection.

# 2024-07-20
## ADDED
- `use_rule_based_filter` config option, implemented in `main.py`
- model_config loaded into make_model_card to access the above config option
- Updated sleep_functions.modify_scores(), but it is not currently working!
