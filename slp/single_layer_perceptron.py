import random

def train_slp(X,y,epochs,learning_rate):
    W=[]
    for i in range(len(X["Points"][0])):
        W.append(random.uniform(-1, 1))
    bias=random.uniform(-1, 1)
    for i in range(epochs):
        errors=0
        for j in range(len(X["Points"])):
            t=0
            for k in range(len(W)):
                t+=X["Points"][j][k]*W[k]
            t+=bias
            h=1 if t>=0 else 0
            error=y[j]-h
            for i in range(len(W)):
                W[i]+=learning_rate*error*X["Points"][j][i]
            bias+=learning_rate*error
            errors+=abs(error)
        if errors==0:
            break
    return W,bias


X = [
    [0,0,0,0],
    [0,0,0,1],
    [0,0,1,0],
    [0,0,1,1],
    [0,1,0,0],
    [0,1,0,1],
    [0,1,1,0],
    [0,1,1,1],
    [1,0,0,0],
    [1,0,0,1],
    [1,0,1,0],
    [1,0,1,1],
    [1,1,0,0],
    [1,1,0,1],
    [1,1,1,0],
    [1,1,1,1]
]

y = [0, 0, 0, 0, 0, 0, 0, 0,
     0, 0, 0, 0, 0, 1, 1, 1]

print(train_slp({"Points": X}, y, 1000, 0.1))