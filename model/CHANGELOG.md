# 2025-01-03
## TODO
- Issue: features are generated at run-time in Mora, w/ hardcoded feature selection. Leads to incompatibility between model artifacts and Mora features.
  - Solution: Change data_processing into feature store and track it w/ versioning. In Mora, reference relevant feature store version for feature selection.
- Make generating the model cards a config option
- Validations: eg validate monotonic increasing epoch if using for time series index

## 2025-01-07
- [FIXED] Rule based filter now working
- [ADDED] test script for rule based filter

## 2025-01-03
### FIXED
- Fixed filename parsing for txt files.

## 2024-07-20
### ADDED
- `use_rule_based_filter` config option, implemented in `main.py`
- model_config loaded into make_model_card to access the above config option
- Updated sleep_functions.modify_scores(), but it is not currently working!
