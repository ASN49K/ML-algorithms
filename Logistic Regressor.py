def sigmoid(z):
    return np.where(z >= 0,
                    1 / (1 + np.exp(-z)),
                    np.exp(z) / (1 + np.exp(z)))
def sigmoid_2(z):
    return 1/(1+np.exp(-z))

class Logistic_regressor:
    def get_z(self,X):
        return X@self.w+self.b
    def probability(self,X):
        return sigmoid(self.get_z(X))
    def predict(self,X):
        return (sigmoid(self.get_z(X)) >= 0.5).astype(int)
    def calculate_gradient_w(self,X,y):
        v=[]
        p=self.probability(X)
        diff=y-p
        return X.T@diff
    def likelihood(self,X,y):
        result=1
        p=self.probability(X)
        for i in range(self.n):
            result*=p[i]**(y[i])
            result*=(1-p[i])**(1-y[i])
        return result
    def log_likelihood(self,X,y):
        result=0
        p=self.probability(X)
        for i in range(self.n):
            result+=np.log(p[i]**(y[i]))
            result+=np.log((1-p[i])**(1-y[i]))
        return result
    def calculate_gradient_b(self,X,y):
        p=self.probability(X)
        return y.sum()-p.sum()
    def __init__(self,X_train,y_train,learning_rate=1e-4):
        np.random.seed(69)
        X=np.array(X_train)
        y=np.array(y_train)
        self.n,self.m=X.shape
        #w=np.random.uniform(-1,1,self.m)
        #b=np.random.uniform(-1,1)
        self.w=np.random.uniform(0,0,self.m)
        self.b=np.random.uniform(0,0)
        z=self.predict(X)
        print(z)
        print(self.w)
        print(self.b)
        print(accuracy_score(z , y))
        ok=0
        last=-1
        it=0
        self.losses=[]
        self.acc=[]
        while it<10000:
            it+=1
            grad_w=self.calculate_gradient_w(X,y)
            grad_b=self.calculate_gradient_b(X,y)
            self.w+=learning_rate*grad_w
            self.b+=learning_rate*grad_b
            z=self.predict(X)
            acc=accuracy_score(z , y)
            ok=(last==acc)
            last=acc
            self.losses.append(self.log_likelihood(X , y))
            self.acc.append(accuracy_score(z , y))
            if it%1000==1:
                print("accuracy is: ", accuracy_score(z , y))
                print("likelihood is: ",self.log_likelihood(X , y))
                #self.losses.append(self.log_likelihood(X , y))
                #self.acc.append(accuracy_score(z , y))
                print("w is: ",self.w)
                print("b is: " , self.b)
