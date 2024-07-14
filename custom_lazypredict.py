from lazypredict.Supervised import LazyClassifier as OriginalLazyClassifier, LazyRegressor as OriginalLazyRegressor
from sklearn.preprocessing import OneHotEncoder

# Monkey patch OneHotEncoder in lazypredict
import lazypredict.Supervised
lazypredict.Supervised.OneHotEncoder = lambda **kwargs: OneHotEncoder(sparse_output=False, **{k: v for k, v in kwargs.items() if k != 'sparse'})

class CustomLazyClassifier(OriginalLazyClassifier):
    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None, predictions=False, random_state=42):
        super().__init__(verbose, ignore_warnings, custom_metric, predictions, random_state)

class CustomLazyRegressor(OriginalLazyRegressor):
    def __init__(self, verbose=0, ignore_warnings=True, custom_metric=None, predictions=False, random_state=42):
        super().__init__(verbose, ignore_warnings, custom_metric, predictions, random_state)