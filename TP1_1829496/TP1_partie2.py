import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X = digits.data
y = digits.target

y_one_hot = np.zeros((y.shape[0],len(np.unique(y))))
y_one_hot[np.arange(y.shape[0]),y] = 1 # one hot target or shape N x K

X_train,X_test,y_train,y_test = train_test_split(X,y_one_hot,test_size = 0.3,random_state = 42)

X_test,X_validation,y_test,y_validation = train_test_split(X_test,y_test,test_size = 0.5,random_state = 42)

W = np.random.normal(0,0.01,(len(np.unique(y)),X.shape[1])) # weights of shape K x L


best_W = None
best_accuracy = 0
lr = 0.0001
nb_epochs = 500
# minibatch_size = len(y)//20
minibatch_size = 1000

losses = []
losses_validation = []
accuracies = []

def softmax(x):
    # assurez vous que la fonction est numeriquement stable
    # e.g. softmax(np.array([1000,10000,100000],ndim = 2))
    return np.exp(x - np.max(x, axis=0)) / np.sum(np.exp(x - np.absolute(np.max(x, axis=0))), axis=0)


def get_accuracy(X,y,W):
    sum = 0
    for i in range(len(X)):
        y_pred = softmax(np.dot(W, X[i]))
        sum = sum + np.vdot(y[i], y_pred)
    return sum / len(X)

def get_grads(y,y_pred,X):
    return -(y-y_pred).T.dot(X) / len(y)


def get_loss(y,y_pred):
    # Using mean absolute error
    loss = -y*np.log(y_pred) - (1-y)*np.log(1-y_pred)
    loss = loss.sum() / len(y)
    return loss

            


for epoch in range(nb_epochs):
    loss = 0
    loss_validation = 0
    accuracy = 0

    #Shuffle
    permutation = list(np.random.permutation(X_train.shape[0]))
    shuffled_X = X_train[permutation, :]
    shuffled_Y = y_train[permutation, :]

    for i in range(0,X_train.shape[0],minibatch_size):
        y_pred = softmax(W.dot(shuffled_X[i:i+minibatch_size,:].T)).T

        grad = get_grads(shuffled_Y[i:i+minibatch_size,:], y_pred, shuffled_X[i:i+minibatch_size,:])
        W = W - lr * grad
    
    y_train_pred = softmax(W.dot(X_train.T)).T

    loss = get_loss(y_train, y_train_pred)

    losses.append(loss) # compute the loss on the train set

  
    accuracy = get_accuracy(X_validation, y_validation, W)
    accuracies.append(accuracy) # compute the accuracy on the validation set
    if accuracy > best_accuracy:
        # select the best parameters based on the validation accuracy
        best_W = W

    y_validation_pred = softmax(W.dot(X_validation.T)).T
    loss_validation = get_loss(y_validation, y_validation_pred)
    losses_validation.append(loss_validation)



accuracy_on_unseen_data = get_accuracy(X_test,y_test,best_W)
print(accuracy_on_unseen_data) # 0.897506925208


ax1 = plt.subplot(221)
ax1.plot(losses, 'blue', losses_validation, 'orange')
ax1.set_ylabel('Average negative log likelihood')
ax1.set_xlabel('Epoch')
ax1.text(200, 1.5, 'Train', color='blue')
ax1.text(200, 1.5, 'Validation', color='orange')
ax3 = plt.subplot(222)
ax3.imshow(best_W[4, :].reshape(8, 8))
ax3.set_title('Poids appris pour chiffre 4')
plt.show()