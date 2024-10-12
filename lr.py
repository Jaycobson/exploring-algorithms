import numpy as np

class LinearRegression:

    def __init__(self):
        self.intercept = None
        self.coef = None

    def fit(self,X,y):
        ones = np.ones((len(X),1))
        X = np.concatenate((ones, X), axis = 1)
        XT = X.T
        XTX = XT.dot(X)
        XTX_inv = np.linalg.inv(XTX)
        XTy = XT.dot(y)
        self.coef = XTX_inv.dot(XTy)

    def predict(self,X):
        ones =  np.ones((len(X), 1))
        X = np.concatenate((ones, X), axis =1)
        return X.dot(self.coef)
    
    def rsquared(self,X,y):
        ypred = self.predict(X)
        ss_total = np.sum((y - np.mean(y))**2)
        ss_residual = np.sum((y - ypred)**2)
        return 1 - (ss_residual / ss_total)
    
lr = LinearRegression()
lr.fit([[2,2,3],[1,3,4], [4,2, 5]],[3, 7 ,5])

pred = lr.predict([[3,5,3]])
print(pred)

# print(lr.rsquared([3, 7 ,5], lr.predict([[2,2,3],[8,9,7], [1,2, 5]])))


