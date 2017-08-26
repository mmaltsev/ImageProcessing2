#task 2. Tensor LDA
import os
import numpy as np
import scipy.misc as msc
from scipy.linalg import inv
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score
from sklearn.metrics import auc
from re import sub
TRAIN_FOLDER = 'uiucTrain/'
TEST_FOLDER = 'uiucTest/'

def getLabels(folder=TRAIN_FOLDER):
    files = os.listdir(folder)
    labels = ['Pos' in name for name in files]
    labels = np.array(labels) * 2 - 1
    return files, labels.astype(float)



# In[46]:

def read_and_center(folder, files):
    X = np.stack([msc.imread(folder + '/' + f) for f in files], axis=0)
    shape = X.shape[1:]
    #X = [image.flatten() for image in X]
    mean = np.mean(X)
    X_center = (X - mean)
    return X_center, mean, shape

def dist(x,y):
    return np.sqrt(np.sum(np.power(x-y, 2)))

def tLDA (eps  = 0.0001):
    #get files
    train_names, y = getLabels(TRAIN_FOLDER)

    for lbl in set(y):
        y[y==lbl] = y[y==lbl] /np.sum(y[y==lbl] )
    #get train data, X - tensor, containing images
    X, meanT, shape =read_and_center(TRAIN_FOLDER, train_names)
    u_ = np.random.rand(1, shape[0])
    v_ = np.random.rand(1, shape[1])
    d = np.inf
    while d > eps:
        u_old = u_
        X_ = np.tensordot(u_.flatten(), X, axes=(0,1))
        v_ = inv( np.dot(X_.T,X_)).dot( X_.T ).dot(y)
        v_old = v_
        X_ = np.tensordot(X, v_, axes=(2,0))
        u_ = inv(np.dot(X_.T, X_)).dot(X_.T).dot(y)
        d = dist(u_,u_old)

    W = np.outer(u_,v_.T)
    #W /= np.max(W)
    return(W, meanT, X)

def predict(W, X,  theta):
    Y_ = np.tensordot(X, W, axes = ([1,2], [0,1] ))
    Yout = np.ones(Y_.shape)
    Yout[Y_ < theta] = -1
    return(Yout)


def estimateModel(W, X, y):
    ##calculate ROC: presison, recall.
    thetas = np.arange(-0.004, 0.006, 0.0002)
    accuracy =[]
    precision = []
    recall = []
    for theta in thetas:
        Ypred = predict(W, X, theta)
        #print(len(Ypred[Ypred>0]))
        accuracy.append( accuracy_score(y, Ypred))
        precision.append( precision_score(y, Ypred))
        recall.append( recall_score(y, Ypred))
    import matplotlib.pyplot as plt
    plt.plot(recall, precision, label='Precision-Recall curve')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.ylim([0.0, 1.05])
    plt.xlim([0.0, 1.0])
    plt.title('Precision-Recall curve, based on theta')
    # Empirically we want the value of recall being ~ 0.725
    plt.axvline(x = 0.725,color = "red")
    plt.show()
    i = np.where(np.abs(np.array(recall) - 0.725) < 0.01)[0]

    return thetas[i], precision[i], recall[i], accuracy[i]

#sliding window test
def slidingWindow(f, W, meanT, threshold):
    img = msc.imread(TEST_FOLDER + '/' + f)
    import matplotlib.pyplot as plt
    #img2 = np.pad(img,(90,90),'constant', constant_values=(0,0))
    plt.imshow(img, cmap="Greys_r")
    ct = plt.gca()
    ct.set_axis_off()
    ct.set_title('image')
    h,w = img.shape # 35, 81

    Xtest = [img[j :j + W.shape[0], i:i + W.shape[1]] for j in np.arange(h-W.shape[0], step = 2) for i in np.arange(w - W.shape[1], step = 2)]

    coord = [(i,j) for j in np.arange(h-W.shape[0], step = 2) for i in np.arange(w - W.shape[1], step = 2)]
    Ypred = predict(W, np.array(Xtest)-meanT, threshold)
    for coords in np.array(coord)[np.where(Ypred == 1.)[0]]:
        rect = plt.Rectangle(coords, W.shape[1], W.shape[0], edgecolor='r', facecolor='none')
        ct.add_patch(rect)

    #  # highlight matched region
    plt.autoscale(False)
    plt.show()
    plt.savefig("task2/out/"+sub("\.pgm", "", f) + "_detection.jpg" )
    plt.close()
def main():
    #Learn model
    W, meanT, X = tLDA(eps = 0.0001)

    #aquire test data
    test_names, _= getLabels(TEST_FOLDER )

    #aquire train labels
    train_names, y = getLabels(TRAIN_FOLDER)

    theta_best, prec, rec, acc = estimateModel(W, X, y)

    for img in test_names[16:30]:
        slidingWindow(img, W,meanT, 0.0035)

