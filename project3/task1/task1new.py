import os
import numpy as np
import scipy.misc as msc
import matplotlib.pyplot as plt
from sklearn.metrics import precision_recall_curve, auc, accuracy_score, precision_score, recall_score

TRAIN_FOLDER = 'uiucTrain/'
TEST_FOLDER = 'uiucTest/'

def read_and_center(folder, files):
    X = np.stack([msc.imread(folder + '/' + f).flatten() for f in files], axis=0)
    shape = msc.imread(folder + '/' + files[0]).shape
    #X = [image.flatten() for image in X]
    mean = np.mean(X)
    X_center = (X - mean)
    return X_center, mean, shape


#get train folder file names and 1,0 class variables
def getLabels(folder=TRAIN_FOLDER):
    files = os.listdir(folder)
    labels = ['Pos' in name for name in files]
    labels = np.array(labels) * 2 - 1
    return files, labels.astype(float)

def predict (w, X_center, theta):
    Yprime = np.dot(X_center, w)
    return ((Yprime > theta) *2 -1)

def estimateModel(w, X_center, y,Xmean,recall_sc):
    ##calculate ROC: presison, recall.
    #val1, val2 = [np.mean(X_center[y ==yi]) for yi in Xmean.keys()]
    val1, val2 = -1, 3
    thetas = np.arange(val1, val2 , abs(val1 - val2 )/10)
    accuracy =[]
    precision = []
    recall = []
    for theta in thetas:
        Ypred = predict(w, X_center, theta)
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
    # Empirically we want the value of recall being ~ 0.9
    #plt.axvline(x = recall_sc,color = "red")
    plt.show()
    i = np.argmin(np.abs(np.array(recall) - recall_sc))

    # create bar plot
    fig, ax = plt.subplots()
    index = np.arange(thetas.__len__())
    bar_width = 0.35
    opacity = 0.8

    rects1 = plt.bar(index, precision, bar_width,
                     alpha=opacity,
                     color='b',
                     label='pr')

    rects2 = plt.bar(index + bar_width, recall, bar_width,
                     alpha=opacity,
                     color='g',
                     label='rec')

    plt.xlabel('Tetta')
    plt.ylabel('')
    plt.title('')
    plt.xticks(index + bar_width, [str(theta)[:5] for theta in thetas])
    plt.legend()

    plt.tight_layout()
    plt.show()

    return thetas[i], precision[i], recall[i], accuracy[i]

def slidingWindow(f, W, meanT, threshold):
    img = msc.imread(TEST_FOLDER + '/' + f)
    import matplotlib.pyplot as plt
    from re import sub
    #img2 = np.pad(img,(90,90),'constant', constant_values=(0,0))
    plt.imshow(img, cmap="Greys_r")
    ct = plt.gca()
    ct.set_axis_off()
    ct.set_title('image')
    h,wid = img.shape # 35, 81

    Xtest = [img[j :j + W.shape[0], i:i + W.shape[1]].flatten() for j in np.arange(h-W.shape[0], step = 2) for i
             in np.arange(wid  - W.shape[1], step = 2)]

    coord = [(i,j) for j in np.arange(h-W.shape[0], step = 2) for i in np.arange(wid  - W.shape[1], step = 2)]
    #predict using precomputed W and X - mu_train
    Ypred = predict(W.flatten(), np.array(Xtest)-meanT, threshold)
    for coords in np.array(coord)[np.where(Ypred == 1.)[0]]:
        rect = plt.Rectangle(coords, W.shape[1], W.shape[0], edgecolor='r', facecolor='none')
        ct.add_patch(rect)

    #  # highlight matched region
    plt.autoscale(False)
    plt.show()
    plt.savefig("task1/out/"+sub("\.pgm", "", f) + "_detection.jpg" )
    plt.close()
def main():
    #read data
    train_names, y = getLabels(TRAIN_FOLDER)
    X_center, meanT, shape = read_and_center(TRAIN_FOLDER, train_names)
    #compute means
    Xmean = dict()
    for yi in set(y):
        Xmean[yi] =  np.mean(X_center[y == yi ] , axis=0)
    #compute scatter matrices
    ## S_k = sum (x - mu_k) (x-mu_k)T
    Sk = dict()
    for yi in set(y):
        Sk[yi] = np.dot((X_center[y == yi] - Xmean[yi]).T,(X_center[y == yi] - Xmean[yi])  )

    ## S_w = S1 + S2
    Sw = Sk[1.0] + Sk[-1.0]
    ## w = S_w ^-1 (mu1 - mu2)
    w = np.linalg.inv( Sw).dot(Xmean[1.0] - Xmean[-1.0])



    test_names, _ = getLabels(TEST_FOLDER)

    W = w.reshape(shape)
    plt.imshow(W, 'Greys')
    plt.savefig("task1/W.jpg")
    for img in test_names[1:15]:
        #theta 2
        slidingWindow(img, W, meanT, 4)


