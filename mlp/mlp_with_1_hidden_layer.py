import random
import math
def sigmoid(x):
    return 1 / (1 + math.exp(-x))
def train_mlp(X,y,epochs,learning_rate):
    W1=[]
    k=int(input("Enter the number of hidden layers points: "))
    for i in range(len(X["Points"][0])):
        W1.append([])
        for j in range(k):
            W1[i].append(random.uniform(-1, 1))
    W2=[]
    for i in range(k):
        W2.append(random.uniform(-1, 1))
    bias=[]
    for i in range(k):
        bias.append(random.uniform(-1, 1))
    bias2=random.uniform(-1, 1)
    for i in range(epochs):
        for j in range(len(X["Points"])):
            z1=[]
            for p in range(k):
                z1.append(sum([X["Points"][j][i] * W1[i][p] for i in range(len(X["Points"][j]))]) + bias[p])
            h1=[]
            for i in range(len(z1)):
                h1.append(sigmoid(z1[i]))
            z2=sum([h1[i]*W2[i] for i in range(k)]) + bias2
            h2=sigmoid(z2)
            #computing loss
            l=-(y[j]*math.log(h2)+(1-y[j])*math.log(1-h2))
            del2=h2-y[j]
            del1=[]
            for i in range(k):
                del1.append(del2*W2[i]*h1[i]*(1-h1[i]))
            for i in range(k):
                W2[i]-=learning_rate*del2*h1[i]
            bias2=bias2-learning_rate*del2
            w1=[]
            for i in range(len(W1)):
                w1.append([])
                for p in range(len(W1[i])):
                    w1[i].append(W1[i][p]-learning_rate*del1[p]*X["Points"][j][i])
            temp_bias=[]
            for i in range(len(bias)):
                temp_bias.append(bias[i]-learning_rate*del1[i])
            W1=w1
            bias=temp_bias
    return W1,W2,bias,bias2