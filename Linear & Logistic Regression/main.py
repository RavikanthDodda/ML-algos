import scipy.io
from math import sqrt,exp,pi

# Returns mean of X where X is a list.
def mean(X):
    return sum(X)/len(X)

# Returns standard deviation of X where X is a list and m is the mean of X.
def sd(X,m):
    return sqrt(sum((i-m)**2 for i in X)/(len(X)))

# Returns re-formatted trX and tsX with mean and standard deviation as features, argument X is either tsX or trX.
def mean_sd(X):
    X_new = []
    for i in range(len(X)):
        column = []
        column.append(mean(X[i]))
        X_new.append(column)

    for i in range(len(X)):
        X_new[i].append(sd(X[i],X_new[i][0]))
    return X_new

# Calculates the probability p(x|y) from feature x, it's mean m and standard deviation sd.
def conditional_probabilty(x, mean, sd):
	return (1 / (sqrt(2 * pi) * sd)) * exp(-((x-mean)**2 / (2 * sd**2 )))

Numpyfile = scipy.io.loadmat("mnist_data.mat") 

trX = Numpyfile['trX']
trY = Numpyfile['trY']
tsX = Numpyfile['tsX']
tsY = Numpyfile['tsY']

# Reformatted training and testing sets with mean and S.D as features.
trX_new = mean_sd(trX)
tsX_new = mean_sd(tsX)

# Seperated training dataset by sample type.
trX_7 = trX_new[:6265]
trX_8 = trX_new[6265:]

# MLE, DE
print("\n+-------Density Estimation-------+")
# parameters(mean,standard deviation) of 2-D normal distribution of features of 7.
mean_7 = [sum(i[0] for i in trX_7)/len(trX_7), sum(i[1] for i in trX_7)/len(trX_7)]
sd_7 = [sd([i[0] for i in trX_7],mean_7[0]),sd([i[1] for i in trX_7],mean_7[1])]

# parameters(mean,standard deviation) of 2-D normal distribution of features of 8.
mean_8 = [mean([i[0] for i in trX_8]),mean([i[1] for i in trX_8])]
sd_8 = [sd([i[0] for i in trX_8],mean_8[0]),sd([i[1] for i in trX_8],mean_8[1])]

# Calculating Covariance matrix
covaraince_matrix = [[sd([i[0] for i in trX_new],mean([i[0] for i in trX_new]))**2,0],[0,sd([i[1] for i in trX_new],mean([i[1] for i in trX_new]))**2]]

print("\nMean of feature 1 (mean) of training set 7 -",mean_7[0])
print("Mean of feature 2 (standard deviation) of training set 7 -",mean_7[1])
print("Mean of feature 1 (mean) of training set 8-",mean_8[0])
print("Mean of feature 2 (standard deviation) of training set 8 -",mean_8[1])

print("\nStandard deviation of feature 1 (mean) of training set 7 -",sd_7[0])
print("Standard deviation of feature 2 (standard deviation) of training set 7 -",sd_7[1])
print("Standard deviation of feature 1 (mean) of training set 8 -",sd_8[0])
print("Standard deviation of feature 2 (standard deviation) of training set 8 -",sd_8[1])

print("\nCovariance matrix is",covaraince_matrix)
# Naive Bayes Classification
print("\n+-------Naive Bayes Classification-------+")

# Prior probabilities of y = 7 and y = 8.
prob_7 = 6265/12116
prob_8 = 1 - prob_7

predictons = []

#calculaing p(x|Y) for y = 7 and y = 8.
for i in tsX_new:
    prob_X_7 = prob_7 * conditional_probabilty(i[0],mean_7[0],sd_7[0]) * conditional_probabilty(i[1],mean_7[1],sd_7[1])
    prob_X_8 = prob_8 * conditional_probabilty(i[0],mean_8[0],sd_8[0]) * conditional_probabilty(i[1],mean_8[1],sd_8[1])
    val = 1
    if prob_X_7>prob_X_8:
        val = 0
    predictons.append(val)

#Calculating accuracies.
accuracy_7 = 0
len_7 = 1028
accuracy_8 = 0
accuracy = 0

for i in range(len(predictons)):
    if(predictons[i]==tsY[0][i]):
        accuracy+=1
        if tsY[0][i]==0:
            accuracy_7+=1
        else:
            accuracy_8+=1

accuracy_7 /= len_7
accuracy_8 /= len(predictons) -  len_7
print("\nAccuracy of 7: %.1f,\nAccuracy of 8: %.1f,\nFinal accuracy: %.1f." %(accuracy_7 * 100,accuracy_8 * 100,accuracy*100/len(predictons)))


#Logistic Regression
print("\n+-------Logistic Regression-------+")

#sigmoid function of logistic regression. 
def sigmoid(weights, X):
    y = weights[0]
    for i in range(len(X)):
        y += weights[i+1]*X[i]
    return 1/(1 + exp(-y))

# Gradient ascent logic. Takes number of iterations, learning rate, training dataset and training labels as arguments respectively.
# Returns calculated weights.
def gradient_ascent(iter, lrate, X, Y):
    weights = [0 for i in range(3)]
    for k in range(iter):
        for j in range(len(X)):
            yh = sigmoid(weights, X[j])
            diff = (Y[0][j] - yh)
            weights[0] += lrate * diff
            for i in range(len(X[j])):
                weights[i+1] += lrate * diff  * X[j][i]
    return weights

# Calculating weights for sigmoid function.Using number of iterations as 1000 and learning rate as 0.0001.
weights = gradient_ascent(1000,0.0001,trX_new,trY)

print("\nWeights:",weights)
# Calculating accuracies.
acc_7 = 0
acc_8 = 0
acc = 0
predictons_lg = []
for i in range(len(tsX_new)):
    p = sigmoid(weights, tsX_new[i])
    if(p < 0.5):
        predictons_lg.append(0)
    else:
        predictons_lg.append(1)

for i in range(len(predictons_lg)):
    if(predictons_lg[i]==tsY[0][i]):
        acc+=1
        if tsY[0][i]==0:
            acc_7+=1
        else:
            acc_8+=1

acc_7 /= len_7
acc_8 /= (len(predictons_lg)-len_7)
print("\nAccuracy of 7: %.1f,\nAccuracy of 8: %.1f,\nFinal accuracy: %.1f." %(acc_7 * 100,acc_8 * 100,acc*100/len(predictons)))

