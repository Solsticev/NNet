import numpy as np
from sklearn.preprocessing import OneHotEncoder
import dataTrans
import matplotlib.pyplot as plt

def sigmoid(x):
    return 1 / (1 + np.exp(-x))
    
def mse_loss(y_true, y_pred):
    return np.square(np.subtract(y_true, y_pred)).mean()

def accuracy(y_true, y_pred):
    return np.mean(y_true == y_pred)

def to_onehot(y):
    # 输入为向量，转为onehot编码
    y = y.reshape(-1, 1)
    enc = OneHotEncoder()
    enc.fit(y)
    y = enc.transform(y).toarray()
    return y

def soft_max(X):
    reg = np.sum(np.exp(X), axis=1)
    return np.exp(X) / reg.reshape(-1, 1)
    
def cross_entropy(y_true, p):
    # y_true: 60000*10 经过one-hot编码  
    # p: 60000*10
    return np.sum(np.multiply(y_true, -np.log(p)), axis=1).mean()

class NeuralNetwork():
    def __init__(self, d, q, l, init=True):
        # weights
        self.v = np.random.randn(d, q)
        self.w = np.random.randn(q, l)
        if init == True:
            self.v = np.random.normal(loc=0, scale=4*np.sqrt(2/(d+q)), size=(d,q))
            self.w = np.random.normal(loc=0, scale=4*np.sqrt(2/(l+q)), size=(q,l))
        # biases
        self.gamma = np.random.randn(q)
        self.theta = np.random.randn(l)
        if init == True:
            self.gamma = np.zeros(q)
            self.theta = np.zeros(l)

    def predict(self, X):
        '''
        X: shape (n_samples, d)
        '''
        b = sigmoid(np.dot(X, self.v) - self.gamma)
        output = soft_max(np.dot(b, self.w) - self.theta)
        return output

    
    def train(self, X, y, X_test, y_test, learning_rate = 0.0005, epochs = 400, reg="L1", err='cross_entropy', update='none'):
        '''
        X: shape (n_samples, d)
        y: shape (n_samples, l)
        输入样本和训练标记，进行网络训练
        '''
        Loss = []
        Acc = []
        count = 0
        L = learning_rate / 10
        if err == 'cross_entropy':
            for epoch in range(epochs):
                if update == 'CLR':
                    if count <= 4:
                        learning_rate += L
                        count += 1
                    elif count > 4 and count <= 9:
                        learning_rate -= L
                        count += 1
                        if count == 10:
                            count = 0
                else:
                    pass
                
                b = sigmoid(np.dot(X, self.v) - self.gamma)
                y_pred = soft_max(np.dot(b, self.w) - self.theta)
    
                # g = np.multiply(y_pred, np.multiply(1 - y_pred, y - y_pred))
                g = y_pred - y
                a = np.dot(self.w, g.T)
                e = b * (1 - b) * a.T
                
                # 计算epoch的loss
                y_preds = self.predict(X)
                loss = cross_entropy(y, y_preds)
                # print("Epoch %d loss: %.3f"%(epoch, loss), flush=True)
                print("Epoch %d loss: %f"%(epoch, loss), flush=True)
                Loss.append(loss)
                
                # 计算测试集准确率
                y_preds_test = self.predict(X_test)
                y_pred_class = np.argmax(y_preds_test, axis=1)
                acc = accuracy(y_test, y_pred_class) * 100
                print("Testing Accuracy: {:.3f} %".format(acc), flush=True)
                Acc.append(acc)
                
                for i in range(X.shape[0]):
                    tmp1 = g[i].reshape(1, g.shape[1])
                    tmp2 = e[i].reshape(1, e.shape[1])
                    tmp3 = b[i].reshape(1, b.shape[1])
                    tmp4 = X[i].reshape(1, X.shape[1])

                    # L2正则化
                    if reg == "L2":
                        self.w -= learning_rate * (np.dot(tmp3.T, tmp1) + self.w / X.shape[0])
                        self.theta -= learning_rate * (g[i] + self.theta / X.shape[0])
                        self.v -= learning_rate * (np.dot(tmp4.T, tmp2) + self.v / X.shape[0])
                        self.gamma -= learning_rate * (e[i] + self.gamma / X.shape[0])
                    # L1正则化
                    elif reg == "L1":
                        self.w -= learning_rate * (np.dot(tmp3.T, tmp1) + np.sign(self.w) / X.shape[0])
                        self.theta -= learning_rate * (g[i] + np.sign(self.theta) / X.shape[0])
                        self.v -= learning_rate * (np.dot(tmp4.T, tmp2) + np.sign(self.v) / X.shape[0])
                        self.gamma -= learning_rate * (e[i] + np.sign(self.gamma) / X.shape[0])
                    elif reg == "none":
                        self.w -= learning_rate * np.dot(tmp3.T, tmp1)
                        self.theta -= learning_rate * g[i]
                        self.v -= learning_rate * np.dot(tmp4.T, tmp2)
                        self.gamma -= learning_rate * e[i] + self.gamma
                    else:
                        print("regularzation failed", flush=True)
                        
        elif err == 'mse':
            for epoch in range(epochs):
                b = sigmoid(np.dot(X, self.v) - self.gamma)
                y_pred = soft_max(np.dot(b, self.w) - self.theta)
                
                g = np.multiply(y_pred, np.multiply(1 - y_pred, y - y_pred))
                a = np.dot(self.w, g.T)
                e = b * (1 - b) * a.T
                # 计算epoch的loss
                y_preds = self.predict(X)
                loss = mse_loss(y, y_preds)
                # print("Epoch %d loss: %.3f"%(epoch, loss), flush=True)
                print("Epoch %d loss: %f"%(epoch, loss), flush=True)
                Loss.append(loss)
            
                # 计算测试集准确率
                y_preds_test = self.predict(X_test)
                y_pred_class = np.argmax(y_preds_test, axis=1)
                acc = accuracy(y_test, y_pred_class) * 100
                print("Testing Accuracy: {:.3f} %".format(acc), flush=True)
                Acc.append(acc)
            
                for i in range(X.shape[0]):
                    tmp1 = g[i].reshape(1, g.shape[1])
                    tmp2 = e[i].reshape(1, e.shape[1])
                    tmp3 = b[i].reshape(1, b.shape[1])
                    tmp4 = X[i].reshape(1, X.shape[1])
                    
                    self.w += learning_rate * np.dot(tmp3.T, tmp1)
                    self.theta -= learning_rate * g[i]
                    self.v += learning_rate * np.dot(tmp4.T, tmp2)
                    self.gamma -= learning_rate * e[i] + self.gamma

        return Loss, Acc
            
if __name__ ==  '__main__':
    # 获取数据集，训练集处理成one-hot编码
    path='data'
    X_train, y_train = dataTrans.load_mnist_train(path)
    # 数据归一化
    Min = X_train.min(axis=1).reshape(-1, 1)
    Max = X_train.max(axis=1).reshape(-1, 1)
    X_train = (X_train - Min) / (Max - Min)
    
    X_test, y_test = dataTrans.load_mnist_test(path)

    Min = X_test.min(axis=1).reshape(-1, 1)
    Max = X_test.max(axis=1).reshape(-1, 1)
    X_test = (X_test - Min) / (Max - Min)
    y_train = to_onehot(y_train)

    # 训练网络
    n_features = X_train.shape[1]
    n_hidden_layer_size = 100
    n_outputs = len(np.unique(y_test))
    
    # network2 = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs, init=False)
    # res1, Res1 = network2.train(X_train, y_train, learning_rate = 0.00002, epochs = 100, X_test=X_test, y_test=y_test, reg="none", err='cross_entropy', update='none')
    network = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    res2, Res2 = network.train(X_train, y_train, learning_rate = 0.00002, epochs = 5000, X_test=X_test, y_test=y_test, reg="none", err='cross_entropy', update='none')
    # network3 = NeuralNetwork(d = n_features, q = n_hidden_layer_size, l = n_outputs)
    # res3, Res3 = network.train(X_train, y_train, learning_rate = 0.00002, epochs = 500, X_test=X_test, y_test=y_test, reg="L1")
    # plt.plot(range(100), res1, c='red')
    # plt.plot(range(100), res2, c='blue')
    # plt.legend(['none', 'Xavier'])
    # plt.show()
    # plt.plot(range(100), Res1, c='red')
    # plt.plot(range(100), Res2, c='blue')
    # plt.legend(['none', 'Xavier'])
    # plt.show()

    # 预测结果
    y_pred = network.predict(X_test)
    y_pred_class = np.argmax(y_pred, axis=1)
    mse = cross_entropy(to_onehot(y_test), y_pred)
    print("\nTesting LOSS: {:.3f}".format(mse))
    acc = accuracy(y_test, y_pred_class) * 100
    print("\nTesting Accuracy: {:.3f} %".format(acc))