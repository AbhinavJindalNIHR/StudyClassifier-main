
#load dependencies
import neuralnet as nn
from sklearn.model_selection import train_test_split
#get data
#clean data
#split & scale data
# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X_np, y_np, test_size = 0.25, random_state=42)

# Scale X data
X_train_sc, X_test_sc = scale_data(X_train, X_test)
#convert to float
#load model
model = nn.make_net(10)
model.summary() #look at model summary - arbitrarily 

