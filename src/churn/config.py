MODEL_VERSION = "CH02 v0.1"

# Churn window in days: a customer churns if no purchase occurs in [snapshot, snapshot + H_DAYS).
H_DAYS = 90
H_SENSITIVITY_DAYS = [30, 60, 90]

# Time-ordered design. The validation snapshot tunes hyperparameters; the final model refits on
# train + validation. Test snapshots start at least H_DAYS after the last training label window.
TRAIN_SNAPSHOTS = ["2017-04-01", "2017-07-01", "2017-10-01"]
VALIDATION_SNAPSHOT = "2018-01-01"
TEST_SNAPSHOTS = ["2018-04-01", "2018-06-01"]

N_SEARCH_ITER = 25
RANDOM_STATE = 42
