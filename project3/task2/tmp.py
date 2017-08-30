import numpy as np

def gs_cofficient(v1, v2):
    return np.dot(v2, v1) / np.dot(v1, v1)

def multiply(cofficient, v):
    return map((lambda x : x * cofficient), v)

def proj(v1, v2):
    return multiply(gs_cofficient(v1, v2) , v1)

def gs(X):
    Y = []
    for i in range(len(X)):
        temp_vec = X[i]
        for inY in Y :
            proj_vec = proj(inY, X[i])
            temp_vec = map(lambda x, y : x - y, temp_vec, proj_vec)
        Y.append(temp_vec)
    return Y
def gs(X, norm = True):
    Y = X[0:1,:].copy()
    for i in range(1, X.shape[0]):
        proj = np.diag((X[i,:].dot(Y.T)/np.linalg.norm(Y,axis=1)**2).flat).dot(Y)
        Y = np.vstack((Y, X[i,:] - proj.sum(0)))
    if norm:
        Y = np.diag(1/np.linalg.norm(Y,axis=1)).dot(Y)
    return Y

def gram_schmidt(vectors):
    basis = []
    for v in vectors:
        w = v - np.sum( np.dot(v,b)*b  for b in basis )
        if (w > 1e-10).any():
            basis.append(w/np.linalg.norm(w))
    return np.array(basis)

test = np.array([[3.0, 1.0], [2.0, 2.0]])
test2 = np.array([[1.0, 1.0, 0.0], [1.0, 3.0, 1.0], [2.0, -1.0, 1.0]])

def normalize (X):
    return X/np.max(X)
normalize(GramSchmidt(test2))
print (normalize(np.array(gs(test2))))
print (np.array(gram_schmidt(test2)))
print (np.array(GramSchmidt(test2)))

vectors = test2
A = np.transpose(np.asarray(vectors)).copy()
m, n = A.shape
V = np.zeros((m, n))
for j in np.arange(n):
    v0 = A[:, j]
    v = v0.copy()
    for i in np.arange(j):
        vi = V[:, i]
        if (abs(vi) > tol).any():
            v -= (np.vdot(v0, vi) / np.vdot(vi, vi)) * vi
    V[:, j] = v

import numpy as np

def gramschmidt(A):
    """
    Applies the Gram-Schmidt method to A
    and returns Q and R, so Q*R = A.
    """
    R = np.zeros((A.shape[1], A.shape[1]))
    Q = np.zeros(A.shape)
    for k in range(0, A.shape[1]):
        R[k, k] = np.sqrt(np.dot(A[:, k], A[:, k]))
        Q[:, k] = A[:, k]/R[k, k]
        for j in range(k+1, A.shape[1]):
            R[k, j] = np.dot(Q[:, k], A[:, j])
            A[:, j] = A[:, j] - R[k, j]*Q[:, k]
    return Q, R

def main():
    """
    Prompts for n and generates a random matrix.
    """
    cols = input('give number of columns : ')
    rows = input('give number of rows : ')
    A = np.random.rand(rows, cols)
    A = test2
    Q, R = gramschmidt(A)
    Q
    R
    np.dot(Q.transpose(), Q)
    np.dot(Q, R)

main()