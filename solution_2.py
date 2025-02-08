from sklearn.svm import SVC

def svm_with_diff_c(train_label, train_data, test_label, test_data):
    '''
    Use different value of cost c to train a svm model. Then apply the trained model
    on testing label and data.
    
    The value of cost c you need to try is listing as follow:
    c = [0.01, 0.1, 1, 2, 3, 5]
    Please set kernel to 'linear' and keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE
    print(len(test_data))
    c = [0.01, 0.1, 1, 2, 3, 5]
    Accuracy = [0]*len(c)
    Train_Accuracy = [0]*len(c)
    for i in range(len(c)):
      clf = SVC(C=c[i],kernel = 'linear')
      clf.fit(train_data,train_label)
      pred = clf.predict(test_data)
      Train_Accuracy[i] = 100*clf.score(train_data,train_label)
      Accuracy[i] = sum([1*(pred[i] == test_label[i]) for i in range(len(test_data))])/len(test_data)
      print("For c = %f, No. of support vectors = %d,%d, and Train_Accuracy = %f, Test_Accuracy = %f"%(c[i], clf.n_support_[0], clf.n_support_[1], Train_Accuracy[i], Accuracy[i]*100))
    # print(Train_Accuracy)
    ### END YOUR CODE

def svm_with_diff_kernel(train_label, train_data, test_label, test_data):
    '''
    Use different kernel to train a svm model. Then apply the trained model
    on testing label and data.
    
    The kernel you need to try is listing as follow:
    'linear': linear kernel
    'poly': polynomial kernel
    'rbf': radial basis function kernel
    Please keep other parameter options as default.
    No return value is needed
    '''

    ### YOUR CODE HERE
    k = ['linear', 'poly', 'rbf']
    Accuracy = [0]*len(k)
    Train_Accuracy = [0]*len(k)
    for i in range(len(k)):
      clf = SVC(kernel=k[i])
      clf.fit(train_data,train_label)
      pred = clf.predict(test_data)
      Accuracy[i] = sum([1*(pred[i] == test_label[i]) for i in range(len(test_data))])/len(test_data)
      Train_Accuracy[i] = 100*clf.score(train_data,train_label)
      print("For %s kernel, Number of support vectors = %d,%d, and Train_Accuracy = %f, Test_Accuracy = %f"%(k[i], clf.n_support_[0], clf.n_support_[1], Train_Accuracy[i], Accuracy[i]*100))
      # print("Number of support vectors are",clf.n_support_)
    # print(Train_Accuracy)
    ### END YOUR CODE
