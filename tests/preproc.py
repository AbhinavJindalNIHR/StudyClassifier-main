#import dependencies
from sklearn.preprocessing import MinMaxScaler
#sclaing function

def scale_data(X_train, X_test):
    """Scale data 0-1 based on min and max in training set"""
    
    # Initialise a new scaling object for normalising input data
    sc = MinMaxScaler()

    # Apply the scaler to the training and test sets
    train_sc = sc.fit_transform(X_train)
    test_sc = sc.fit_transform(X_test)
    
    return train_sc, test_sc