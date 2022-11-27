import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.feature_selection import RFE
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('PATH/TO/DATA/FILE', sep='COLUMN_SEPARATOR', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

# Average CV score on the training set was: 0.7857142857142858
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=SGDClassifier(alpha=0.01, eta0=0.1, fit_intercept=False, l1_ratio=0.75, learning_rate="invscaling", loss="squared_hinge", penalty="elasticnet", power_t=1.0)),
    RFE(estimator=ExtraTreesClassifier(criterion="gini", max_features=1.0, n_estimators=100), step=0.55),
    PCA(iterated_power=5, svd_solver="randomized"),
    SGDClassifier(alpha=0.0, eta0=0.1, fit_intercept=False, l1_ratio=0.5, learning_rate="constant", loss="log", penalty="elasticnet", power_t=0.0)
)
# Fix random state for all the steps in exported pipeline
set_param_recursive(exported_pipeline.steps, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
