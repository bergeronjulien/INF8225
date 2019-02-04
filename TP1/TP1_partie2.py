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
lr = 0.01
nb_epochs = 500
# minibatch_size = len(y)//20
minibatch_size = 1000

losses = []
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
    return loss.sum() / len(y)


for epoch in range(nb_epochs):
    loss = 0
    accuracy = 0
    grad = 0
    
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

accuracy_on_unseen_data = get_accuracy(X_test,y_test,best_W)
print(accuracy_on_unseen_data) # 0.897506925208

plt.plot(losses)

plt.imshow(best_W[4,:].reshape(8,8))

plt.figure(1)
ax1 = plt.subplot(221)
ax1.plot(losses, 'b')
ax1.set_title('Courbes d\'apprentissage')
ax1.set_ylabel('Log ngatif de vraisemblance moyenne')
ax1.set_xlabel('Epoch')
ax1.text(30, .55, 'Entranement', color='blue')
ax2 = plt.subplot(222)
ax2.plot(accuracies)
ax2.set_title('Prcision sur l\'ensemble de validation')
ax2.set_ylabel('Pourcentage')
ax2.set_xlabel('Epoch')
# ax1.text(30, .65, 'Validation', color='magenta')
ax3 = plt.subplot(223)
ax3.imshow(best_W[4, :].reshape(8, 8))
ax3.set_title('Poids du chiffre 4')
ax4 = plt.subplot(224)
ax4.imshow(best_W[7, :].reshape(8, 8))
ax4.set_title('Poids du chiffre 7')
plt.show()