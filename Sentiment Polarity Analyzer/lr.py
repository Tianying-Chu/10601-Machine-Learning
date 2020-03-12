# Name: Tianying Chu
# Andrew ID: tianying

# Logistic Regression

import sys
import numpy as np

def readData(file):
    with open(file, 'r') as f:
        data = list() # data stores all the examples
        for line in f: # each line is one example
            y_i = int(line.split()[0])
            x_i = dict()
            for item in line.split()[1:]:
                x_i[item.split(':')[0]] = int(item.split(':')[1]) # store the value of each attributes
                x_i['bias'] = 1 # fold the bias term into x_i
            example = {'y': y_i, 'x': x_i} # example is a dictionary of y_i & x_i
            data.append(example)
        return data


def initialTheta(dict_file):
    with open(dict_file, 'r') as f:
        theta = dict()
        for line in f:
            theta[line.split()[1]] = 0
            theta['bias'] = 0
        return theta

def sparse_dot(x_i, theta): 
    # x_i is one example in sparse representation. {'j':x_i_j}
    # theta is a 'len(dict)+1' * 1 dictionary. {'j': theta_j}
    product = 0.0
    for j, x_i_j in x_i.items():
        if j not in theta.keys():
            continue
        product += theta[j] * x_i_j
    return product

def sigmoid(a):
    return 1 / (1 + np.exp(-a))

def SGD(example_i, theta, step_size):
    y_i = example_i['y']
    x_i = example_i['x']
    dot_product_i = sparse_dot(x_i, theta)
    sigm_i = sigmoid(dot_product_i)
    
    # UPDATE theta for EACH ATTR. 
    # theta -= step_size * gradient_J_i_j
    for j, x_i_j in x_i.items():
        gradient_J_i_j = (sigm_i - y_i) * x_i_j # calculate gradient_J_i_j
        theta[j] -= step_size * gradient_J_i_j
    return theta

def train(data, theta, step_size, num_epoch):
    while num_epoch > 0:
        for example_i in data:
            theta = SGD(example_i, theta, step_size)
        num_epoch -= 1
    return theta

def predict(data, theta_MLE):
    predicted_y = list()
    for example_i in data:
        x_i = example_i['x']
        
        dot_product_i = sparse_dot(x_i, theta_MLE)
        sigm_i = sigmoid(dot_product_i)

        if sigm_i >= 0.5:
            predicted_y.append(1)
        else:
            predicted_y.append(0)
    return predicted_y
        
def errorRate(predicted_y, data):
    error = 0
    for (predicted_y_i, example_i) in zip(predicted_y, data):
        y_i = example_i['y']
        if predicted_y_i != y_i:
            error += 1
    errorRate = error / len(predicted_y)
    return errorRate

def writeLabels(file, predicted_y):
    with open(file, 'w') as f:
        for predicted_y_i in predicted_y:
            f.write(str(predicted_y_i) + '\n')

def writeMetrics(file, train_error, test_error):
    with open(file, 'w') as f:
        f.write('error(train): %.6f\nerror(test): %.6f' % (train_error, test_error))

def main():
    # read in datasets
    train_data = readData(formatted_train_input)
    valid_data = readData(formatted_valid_input)
    test_data = readData(formatted_test_input)
    
    # using SGD to find theta_MLE
    theta = initialTheta(dict_input)
    step_size = 0.1
    theta_MLE = train(train_data, theta, step_size, num_epoch)
    
    # predict y labels
    predicted_y_train = predict(train_data, theta_MLE)
    predicted_y_valid = predict(valid_data, theta_MLE)
    predicted_y_test = predict(test_data, theta_MLE)
    
    # calculate error rate
    train_error = errorRate(predicted_y_train, train_data)
    test_error = errorRate(predicted_y_test, test_data)
    
    # output labels and error rate
    writeLabels(train_out, predicted_y_train)
    writeLabels(test_out, predicted_y_test)
    writeMetrics(metrics_out, train_error, test_error)


if __name__ == '__main__':
    '''
    formatted_train_input = 'formatted_train.tsv'
    formatted_valid_input = 'formatted_valid.tsv'
    formatted_test_input = 'formatted_test.tsv'
    dict_input = 'dict.txt'
    train_out = 'train_out.labels'
    test_out = 'test_out.labels'
    metrics_out = 'metrics_out.txt'
    num_epoch = int('30')
    
    '''
    formatted_train_input = sys.argv[1]
    formatted_valid_input = sys.argv[2]
    formatted_test_input = sys.argv[3]
    dict_input = sys.argv[4]
    train_out = sys.argv[5]
    test_out = sys.argv[6]
    metrics_out = sys.argv[7]
    num_epoch = int(sys.argv[8])
    
    
    main()
