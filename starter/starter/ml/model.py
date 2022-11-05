from sklearn.metrics import fbeta_score, precision_score, recall_score
from sklearn.ensemble import RandomForestClassifier

# Optional: implement hyperparameter tuning.
def train_model(X_train, y_train):
    """
    Trains a machine learning model and returns it.

    Inputs
    ------
    X_train : np.array
        Training data.
    y_train : np.array
        Labels.
    Returns
    -------
    model
        Trained machine learning model.
    """

    model= RandomForestClassifier(max_depth=2, random_state=0)
    model.fit(X_train,y_train)


    return model


def compute_model_metrics(y, preds):
    """
    Validates the trained machine learning model using precision, recall, and F1.

    Inputs
    ------
    y : np.array
        Known labels, binarized.
    preds : np.array
        Predicted labels, binarized.
    Returns
    -------
    precision : float
    recall : float
    fbeta : float
    """
    fbeta = fbeta_score(y, preds, beta=1, zero_division=1)
    precision = precision_score(y, preds, zero_division=1)
    recall = recall_score(y, preds, zero_division=1)
    return precision, recall, fbeta


def inference(model, X):
    """ Run model inferences and return the predictions.

    Inputs
    ------
    model :??
        Trained machine learning model.
    X : np.array
        Data used for prediction.
    Returns
    -------
    preds : np.array
        Predictions from the model.
    """
    prediction =model.predict(X)

    return prediction

def slice_metrics(model,X,Y, slice_col,slice_val):

    """
    A fucntion to calculte the model performance on slice of the data
    
    Inputs
    ------
    model:
        Teainded model
    X: np.array
        Features
    Y: np.array
        targer
    slice_col: str
        A colums to slice the data
    slice_val: str
        A category
    Returns
    _ _ _ _
    precision : float
    recall : float
    fbeta : float
    """

    X=X[X[slice_col]==slice_val]
    Y=Y[X[slice_col]==slice_val]

    preds=inference(model,X)
    precision, recall, fbeta=compute_model_metrics(Y, preds)

    return precision, recall, fbeta