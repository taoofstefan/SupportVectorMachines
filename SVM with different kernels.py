def SVM_linear(n_samples, centers, random_state):
    # SVM classifier with linear kernel
    import matplotlib.pyplot as plt
    from sklearn import svm
    from sklearn.datasets import make_blobs
    from sklearn.inspection import DecisionBoundaryDisplay
    
    # we create 20 separable points
    X, y = make_blobs(n_samples=n_samples, centers=centers, random_state=random_state)
    
    # fit the model, don't regularize for illustration purposes
    clf = svm.SVC(kernel="linear", C=100)
    clf.fit(X,y)
    plt.scatter(X[:, 0], X[:, 1], c=y, s=100, cmap=plt.cm.Paired)
    
    # plot the decision function
    ax = plt.gca()
    DecisionBoundaryDisplay.from_estimator(clf, X, plot_method ="contour", colors="k", level=[-1, 0, 1], alpha=0.5,
                                           linestyles=["-","--","-"], ax=ax)
                                            
    # plot support vectors
    ax.scatter(clf.support_vectors_[:,0], clf.support_vectors_[:,1], s=100, linewidth=1, facecolors="none", edgecolors ="k")
    plt.show()
    
SVM_linear(n_samples=250, centers=2, random_state=42)


def SVM_polynomial(n_samples):
    # SVM regression with poly kernel
    import numpy as np
    from sklearn.svm import SVR
    import matplotlib.pyplot as plt
    X = np.sort(5 * np.random.rand(n_samples, 1), axis=0)
    y = np.sin(X).ravel()
    
    # add noise to targets
    y += 2 * (0.5 - np.random.rand(y.shape[0]))
    svr = SVR(kernel="poly", C=100, gamma="auto", degree=3, epsilon=0.1, coef0=1)
    ax= plt.gca()
    
    #plot regression curve
    ax.plot(X, svr.fit(X, y).predict(X), color="green", lw=1)
    ax.scatter(X[svr.support_], y[svr.support_], facecolor="none", edgecolor="k",s=20)
    plt.show()

SVM_polynomial(n_samples=100)
