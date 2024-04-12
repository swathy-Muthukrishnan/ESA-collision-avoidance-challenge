from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import RandomForestClassifier


class Risk_Forecasting_model:
    """
    A class for training and testing risk forecasting models.

    Methods:
    - __init__(): Initializes an instance of the class.
    - train_risk_predictor(X_train, y_train): Trains a risk predictor model.
    - train_risk_classifier(X_train, y_train): Trains a risk classifier model.
    - test_risk_classifier(X_test, classifier: RandomForestClassifier): Tests a risk classifier model.
    - test_risk_predictor(X_test, predictor:RandomForestRegressor): Tests a risk predictor model.
    """
    def __init__(self):
        pass

    def train_risk_predictor(self, X_train, y_train):
        """
        Trains a risk predictor model.

        Args:
        - X_train: Training features.
        - y_train: Training labels.

        Returns:
        - RandomForestRegressor: Trained risk predictor model.
        """
        model = RandomForestRegressor(n_estimators=100, random_state=42)
        model.fit(X_train, y_train)
        return model
    
    def train_risk_classifier(self, X_train, y_train ):
        """
        Trains a risk classifier model.

        Args:
        - X_train: Training features.
        - y_train: Training labels.

        Returns:
        - RandomForestClassifier: Trained risk classifier model.
        """
        rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
        rf_classifier.fit(X_train, y_train)
        return rf_classifier
    
    def test_risk_classifier(self, X_test, classifier: RandomForestClassifier):
        """
        Tests a risk classifier model.

        Args:
        - X_test: Testing features.
        - classifier (RandomForestClassifier): Trained risk classifier model.

        Returns:
        - array-like: Predicted labels.
        """
        return classifier.predict(X_test)

    def test_risk_predictor(self, X_test, predictor:RandomForestRegressor):
        """
        Tests a risk predictor model.

        Args:
        - X_test: Testing features.
        - predictor (RandomForestRegressor): Trained risk predictor model.

        Returns:
        - array-like: Predicted values.
        """
        return predictor.predict(X_test)
    
