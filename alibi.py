from alibi_detect.cd import DriftKSD
from alibi_detect.utils.data import create_outlier_batch

# Initialize the drift detector
drift_detector = DriftKSD(X_ref=X, p_val=0.05, n_permutations=100)

# Define a function to detect drift
def detect_drift(X_new):
    return drift_detector.predict(X_new)
