# Name: Tianying Chu
# Andrew ID: tianying

# Feature Engineering

import sys

def readData(file):
    data = list()
    with open(file, 'r') as f:
        for line in f:
            y = line.split('\t')[0]
            x = line.split('\t')[1].strip().split(' ')
            example = dict()
            example['y'] = y
            example['x'] = x
            data.append(example)
        return data

def readDict(file):
    vocab = dict()
    with open(file, 'r') as f:
        for line in f:
            word = line.split()[0]
            index = line.split()[1]
            vocab[word] = index
        return vocab

def occur(data, vocab):
    formatted_data = list()
    for example in data:
        x_i = example['x']
        formatted_example = dict()
        formatted_x_i = dict()
        for word in x_i:
            if word in vocab.keys():
                formatted_x_i[vocab[word]] = 1
        formatted_example['y'] = example['y']
        formatted_example['x'] = formatted_x_i
        formatted_data.append(formatted_example)
    return formatted_data

def trim(data, vocab, threshold):
    formatted_data = list()
    for (i, example) in zip(range(len(data)), data):
        x_i = example['x']
        formatted_example = dict()
        formatted_x_i = dict()
        for word in x_i:
            if word in vocab.keys() and data[i]['x'].count(word) < threshold:
                formatted_x_i[vocab[word]] = 1
        formatted_example['y'] = example['y']
        formatted_example['x'] = formatted_x_i
        formatted_data.append(formatted_example)
    return formatted_data
        
def writeData(data, file):
    #print(data[0]['x'], file)
    with open(file, 'w') as f:
        for example in data:
            f.write(example['y'] + '\t')
            for index, value in example['x'].items():
                f.write(index + ':' + str(value) +'\t')
            f.write('\n')

def main():
    train_data = readData(train_input)
    valid_data = readData(validation_input)
    test_data = readData(test_input)
    vocab = readDict(dict_input)
    
    if feature_flag == 1:
        formatted_train_data = occur(train_data, vocab)
        formatted_valid_data = occur(valid_data, vocab)
        formatted_test_data = occur(test_data, vocab)
        
    if feature_flag == 2:
        formatted_train_data = trim(train_data, vocab, threshold)
        formatted_valid_data = trim(valid_data, vocab, threshold)
        formatted_test_data = trim(test_data, vocab, threshold)
        
      
    writeData(formatted_train_data, formatted_train_out)
    writeData(formatted_valid_data, formatted_validation_out)
    writeData(formatted_test_data, formatted_test_out)


if __name__ == '__main__':
    
    '''
    train_input = 'train_data.tsv'
    validation_input = 'valid_data.tsv'
    test_input = 'test_data.tsv'
    dict_input = 'dict.txt'
    formatted_train_out = 'formatted_train.tsv'
    formatted_validation_out = 'formatted_valid.tsv'
    formatted_test_out = 'formatted_test.tsv'
    feature_flag = int('2')
    threshold = 4    
    '''    
    train_input = sys.argv[1]
    validation_input = sys.argv[2]
    test_input = sys.argv[3]
    dict_input = sys.argv[4]
    formatted_train_out = sys.argv[5]
    formatted_validation_out = sys.argv[6]
    formatted_test_out = sys.argv[7]
    feature_flag = int(sys.argv[8])
    threshold = 4
    
    main()

    
    