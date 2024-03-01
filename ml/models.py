import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import confusion_matrix
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, classification_report
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from tqdm import tqdm
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import accuracy_score, classification_report
from sklearn.ensemble import AdaBoostClassifier
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

# Arrays numpy.array are returned, saved to CSV, and later will be divided into chunks
# np.savetxt('X_train_scaled.csv', X_train, delimiter=",")
# np.savetxt('X_test_scaled.csv', X_test, delimiter=",")

# GOOD TRAIN-TEST SPLIT

def train_test_sets_preparation(df):

	# Splitting into training and testing sets while maintaining class proportions
	X_train, X_test, y_train, y_test = train_test_split(df.drop('DELAY', axis=1), df['DELAY'], test_size=0.2, stratify=df['DELAY'], random_state=42)

	# Splitting the training set again into classes 0 and 1
	X_train_0 = X_train[y_train == 0]
	X_train_1 = X_train[y_train == 1]
	y_train_0 = y_train[y_train == 0]
	y_train_1 = y_train[y_train == 1]

	n = min(len(X_train_0), len(X_train_1))

	# Combining subsets of class 0 and 1 in a 1:1 ratio
	X_train_balanced = pd.concat([X_train_0.sample(n=n, random_state=42), X_train_1.sample(n=n, random_state=42)])
	y_train_balanced = pd.concat([y_train_0.sample(n=n, random_state=42), y_train_1.sample(n=n, random_state=42)])

	# X_train_balanced.to_csv('X_train.csv')
	# X_test.to_csv('X_test.csv')
	# y_train_balanced.to_csv('y_train.csv')
	# y_test.to_csv('y_test.csv')

	return X_train_balanced, X_test, y_train_balanced, y_test

#MODELS FOR UNDIVIDED DATA

# just to check accuracy of found grid from searching
def BestModelAccuracy(grid_search, X_test, y_test):
    
	best_params = grid_search.best_params_
	best_model = grid_search.best_estimator_
 
	
	y_pred = best_model.predict(X_test)

	acc = accuracy_score(y_test, y_pred)
	print("Accuracy:", acc)
 
	print(f"Best params: {best_params}")
 
	print('\nClassification Report:')
	print(classification_report(y_test, y_pred))

def Bayesian(X_train, X_test, y_train, y_test):
	# Initializing the GDA classifier
	clf = GaussianNB()

	# Training the classifier on the training data
 
	params = {'var_smoothing': [1e-9, 1e-8, 1e-7, 1e-6, 1e-5]}
 
	grid_search = 	GridSearchCV(clf, params, cv = 5, scoring='accuracy')
	grid_search.fit(X_train, y_train)

	BestModelAccuracy(grid_search, X_test, y_test)
 
 

def SVM(X_train, X_test, y_train, y_test):
    # Initializing the SVM classifier
    clf = SVC()

    # Training the classifier on the training data
    clf.fit(X_train, y_train)

    # Predicting labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluating the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Accuracy: {accuracy:.2f}')

    # Displaying the full classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))


def LogisticREG(X_train, X_test, y_train, y_test):
	# Initializing the logistic regression model
	model = LogisticRegression()

	# Different regression parameters
	param_grid = {'C': [0.001, 0.01, 0.1, 1, 10, 100], 'solver': ['liblinear', 'newton-cg', 'lbfgs', 'sag']}

	# Grid search for finding the best parameters
	grid_search = GridSearchCV(model, param_grid, cv=5, scoring='accuracy')
	grid_search.fit(X_train, y_train)

	# Extracting the best parameters and best estimator from grid search
	best_params = grid_search.best_params_
	best_model = grid_search.best_estimator_

	# Predicting labels for the test set using the best model
	y_pred = best_model.predict(X_test)

	# Calculating the accuracy of the model
	accuracy = accuracy_score(y_test, y_pred)

	# Printing the best parameters and accuracy
	print("LR best parameters:", best_params)
	print("Accuracy:", accuracy)



def DecisionTree(X_train, X_test, y_train, y_test, max_depth=4):
    # Initializing the decision tree classifier
    clf = DecisionTreeClassifier(max_depth=max_depth, random_state=42, criterion='gini')

    # Training the classifier on the training data
    clf.fit(X_train, y_train)

    # Predicting labels for the test data
    y_pred = clf.predict(X_test)

    # Evaluating the classifier's accuracy
    accuracy = accuracy_score(y_test, y_pred)
    print(f'Classification Accuracy: {accuracy:.2f}')

    # Displaying the full classification report
    print('\nClassification Report:')
    print(classification_report(y_test, y_pred))
    print(confusion_matrix(y_test, y_pred))

    # Plotting the decision tree
    #plt.figure(figsize=(12, 8))
    #plot_tree(clf, filled=True, feature_names=X_train.columns, class_names=['Class 0', 'Class 1'])
    #plt.show()

def KNN(X_train, X_test, y_train, y_test):
	# Define the range of values for the parameter k to search
	#param_grid = {'n_neighbors': [3, 5]}

	# Initialize the KNN classifier
	# use metric w minkowsky with euclidean distance
	knn = KNeighborsClassifier(metric = 'minkowski', p = 2, n_neighbors=3)

	knn.fit(X_train, y_train)
	y_pred = knn.predict(X_test)

	accuracy = accuracy_score(y_test, y_pred)
 
	print(f"accuracy: {accuracy}")
	print(classification_report(y_test, y_pred))
 
 	# Create the GridSearchCV object
	#grid_search = GridSearchCV(estimator=knn, param_grid=param_grid, cv=5, scoring='accuracy', verbose = 10)
	# Fit the model to the training data
	#grid_search.fit(X_train, y_train)
	#BestModelAccuracy(grid_search, X_test, y_test)


def SGD(X_train, X_test, y_train, y_test):
	sgdc = SGDClassifier()
	param_grid = {     
		'alpha': [0.0001, 0.001, 0.01],              
		'max_iter': [1000, 2000, 3000, 5000, 10000, 20000],    
	}

	grid_search = GridSearchCV(estimator=sgdc, param_grid=param_grid, cv=5,n_jobs=1, verbose = 10)

	grid_search.fit(X_train, y_train)
	BestModelAccuracy(grid_search, X_test, y_test)


def ADA(X_train, X_test, y_train, y_test):
    clf = AdaBoostClassifier(n_estimators=100, algorithm="SAMME", random_state=0)
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print(f"Accuracy: {accuracy_score(y_test, y_pred)}")
    print(classification_report(y_test, y_pred))



def network(X_train, X_test, y_train, y_test):
    model = Sequential([
        Dense(64, activation='relu', input_shape=(X_train.shape[1],)),  # Hidden layer with 64 neurons and ReLU activation function
        Dense(32, activation='relu'), 
        Dense(16, activation='relu'),  
         
        Dense(8, activation='relu'),   # Second hidden layer with 32 neurons and ReLU activation function
        Dense(1, activation='sigmoid')                             # Output layer with one neuron and sigmoid activation function
    ])

    # Compiling the model
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # Training the model
    history=model.fit(X_train, y_train, epochs=100, batch_size=328, validation_data=(X_test, y_test))
    #i would like to get waights of this model to the file to use it in the future
    model.save_weights('model.h5')
    # Evaluating the model
    loss, accuracy = model.evaluate(X_test, y_test)
    print("Test Loss:", loss)
    print("Test Accuracy:", accuracy)
    return model, history

# MODELS ON SPLIT DATA (live lerning)

# Assumptions are such that from the entire normalized file X_train, X_train_1, ..., X_train_7 were created and similarly from X_test. Each contains a maximum of one million rows. y_train and y_test are whole without divisions.

# It turns out that we need to drop the column with 'datatype' from the Domin dataset
# Drop also for Unnamed: 0
# These drops from above must be done separately for the test and training sets
# and now we can normalize
# in test

X_test = pd.read_csv("/home/meks/Desktop/danexD/X_test.csv")
X_train = pd.read_csv("/home/meks/Desktop/danexD/X_train.csv")
y_test = pd.read_csv("/home/meks/Desktop/danexD/y_test.csv")
y_train = pd.read_csv("/home/meks/Desktop/danexD/y_train.csv")

X_test = X_test.drop(['ARR_DELAY', 'ARR_TIME'], axis=1)
X_train = X_train.drop(['ARR_DELAY', 'ARR_TIME'], axis=1)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

y_train = y_train.values.ravel()
y_test = y_test.values.ravel()

print(y_test.shape)
print(y_train.shape)
print(X_test.shape)
print(X_train.shape)

model = input("give model name: ")

print("begin:")

match model:
    case 'bayes':
        Bayesian(X_train, X_test, y_train, y_test)

    case 'decision':
        # DecisionTree(X_train, X_test, y_train, y_test, max_depth=2)
        # DecisionTree(X_train, X_test, y_train, y_test, max_depth=4)
        # DecisionTree(X_train, X_test, y_train, y_test, max_depth=8)
        DecisionTree(X_train, X_test, y_train, y_test, max_depth=17)

    case 'sgd':
        SGD(X_train, X_test, y_train, y_test)

    case 'knn':
        KNN(X_train, X_test, y_train, y_test)

    case 'gda':
        GDA(X_train, X_test, y_train, y_test)

    case 'forest':
        RandomForest(X_train, X_test, y_train, y_test)

    case 'ada':
        print("ada")
        ADA(X_train, X_test, y_train, y_test)

# SVM(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())

# SVM(X_train, X_test, y_train.values.ravel(), y_test.values.ravel())

