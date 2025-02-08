import numpy as np
import sys
from helper import *


def show_images(data):
    """Show the input images and save them.

    Args:
        data: A stack of two images from train data with shape (2, 16, 16).
              Each of the image has the shape (16, 16)

    Returns:
        Do not return any arguments. Save the plots to 'image_1.*' and 'image_2.*' and
        include them in your report
    """
    ### YOUR CODE HERE
    plt.figure()
    plt.imshow(data[0], cmap='gray')
    images_dir = '/content/drive/MyDrive/HW1/code'
    plt.savefig(f"{images_dir}/image_1.png")
    plt.show()

    plt.figure()
    plt.imshow(data[1], cmap='gray')
    images_dir = '/content/drive/MyDrive/HW1/code'
    plt.savefig(f"{images_dir}/image_2.png")
    plt.show()

    ### END YOUR CODE


def show_features(X, y, save=True):
    """Plot a 2-D scatter plot in the feature space and save it. 

    Args:
        X: An array of shape [n_samples, n_features].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        save: Boolean. The function will save the figure only if save is True.

    Returns:
        Do not return any arguments. Save the plot to 'train_features.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    
    plt.figure()
    for i in range(X.shape[0]):
      if y[i] > 0:
        plt.scatter(X[i,0],X[i,1], s=75, c='red', marker='*')
      else:
        plt.scatter(X[i,0],X[i,1], s=75, c='blue', marker='+')
    images_dir = '/content/drive/MyDrive/HW1/code'
    plt.savefig(f"{images_dir}/train_features.png")
    plt.show()



    ### END YOUR CODE


class Perceptron(object):
    
    def __init__(self, max_iter):
        self.max_iter = max_iter

    def fit(self, X, y):
        """Train perceptron model on data (X,y).

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            self: Returns an instance of self.
        """
        ### YOUR CODE HERE
        w = np.zeros([X.shape[1],1])
        
        i = 0
        n = 0
        while(n<=self.max_iter):

          y_hat = np.sign(w.T@X[i][:,np.newaxis])
          if y [i] != y_hat:
            w += y[i]*X[i][:,np.newaxis]
            # break
          if i > X.shape[0]:
            break
          i += 1
          n += 1
        
        # After implementation, assign your weights w to self as below:
        self.W = w
        
        ### END YOUR CODE
        
        return self

    def get_params(self):
        """Get parameters for this perceptron model.

        Returns:
            W: An array of shape [n_features,].
        """
        if self.W is None:
            print("Run fit first!")
            sys.exit(-1)
        return self.W

    def predict(self, X):
        """Predict class labels for samples in X.

        Args:
            X: An array of shape [n_samples, n_features].

        Returns:
            preds: An array of shape [n_samples,]. Only contains 1 or -1.
        """
        ### YOUR CODE HERE

        return [np.sign(self.W.T@X[i][:,np.newaxis]) for i in range(X.shape[0])]

        ### END YOUR CODE

    def score(self, X, y):
        """Returns the mean accuracy on the given test data and labels.

        Args:
            X: An array of shape [n_samples, n_features].
            y: An array of shape [n_samples,]. Only contains 1 or -1.

        Returns:
            score: An float. Mean accuracy of self.predict(X) wrt. y.
        """
        ### YOUR CODE HERE
        y_hat = self.predict(X)
        # y_hat = np.array(y_hat)
        return np.sum([1*(y_hat[i] == y[i]) for i in range(X.shape[0])])/X.shape[0]
        # return np.average(np.equal(y_hat, y))

        ### END YOUR CODE




def show_result(X, y, W):
    """Plot the linear model after training. 
       You can call show_features with 'save' being False for convenience.

    Args:
        X: An array of shape [n_samples, 2].
        y: An array of shape [n_samples,]. Only contains 1 or -1.
        W: An array of shape [n_features,].
    
    Returns:
        Do not return any arguments. Save the plot to 'result.*' and include it
        in your report.
    """
    ### YOUR CODE HERE
    # print(W[1:3].shape)
    # print(X[1].shape)
    line_value = [W[1:3].T@X[i] for i in range(X.shape[0])]
    x = np.linspace(np.amin(X[:,0]),np.amax(X[:,1]))
    z = -(W[2]/W[1])*x-(W[0]/W[1])
    plt.figure()
    for i in range(X.shape[0]):
      if y[i] > 0:
        plt.scatter(X[i,0],X[i,1], s=75, c='red', marker='*')
      else:
        plt.scatter(X[i,0],X[i,1], s=75, c='blue', marker='+')
      plt.plot(x,z)
    images_dir = '/content/drive/MyDrive/HW1/code'
    plt.savefig(f"{images_dir}/result.png")
    plt.show()

    ### END YOUR CODE



def test_perceptron(max_iter, X_train, y_train, X_test, y_test):

    # train perceptron
    model = Perceptron(max_iter)
    model.fit(X_train, y_train)
    train_acc = model.score(X_train, y_train)
    W = model.get_params()

    # test perceptron model
    test_acc = model.score(X_test, y_test)

    return W, train_acc, test_acc