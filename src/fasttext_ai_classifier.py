import csv, fasttext
import numpy as np

# initialize text and pos_combined data arrays
text_data = []
pos_data = []

# open and read the dataset to gather data
with open('data/training_dataset_subsample.csv', newline= '', encoding= 'utf-8') as csvfile:
    reader = csv.reader(csvfile)
    header = True
    
    for row in reader:
        if header:
            header = False
        else:
            # remove "__label__0 [removed]" data points
            # these are just deleted comments on Reddit that appear often and are all labeled human generated in our dataset
            if row[1] != "[removed]":
                text_data.append('__label__' + row[2] + ' ' + row[1])
            
            pos_data.append('__label__' + row[2] + ' ' + row[4])

# replace new lines with two spaces so there is one entry per line -- improves model performance and data visualization
text_data = [row.replace('\n', '  ') for row in text_data]

# shuffle the data
text_data = np.asarray(text_data)
np.random.shuffle(text_data)
pos_data = np.asarray(pos_data)
np.random.shuffle(pos_data)

# divide into training and test sets
train_split = 0.8 # portion of data that is used for training, test on the rest
split_ind = int(len(text_data) * train_split)
text_train_data = text_data[:split_ind]
text_test_data = text_data[split_ind:]
split_ind = int(len(pos_data) * train_split)
pos_train_data = pos_data[:split_ind]
pos_test_data = pos_data[split_ind:]

# write text files that fasttext models take as input for training and testing 
with open('data/text_train.txt', 'w', encoding= 'utf-8') as txtfile:
    for row in text_train_data:
        txtfile.write(row + '\n')
with open('data/text_test.txt', 'w', encoding= 'utf-8') as txtfile:
    for row in text_test_data:
        txtfile.write(row + '\n')
with open('data/pos_train.txt', 'w', encoding= 'utf-8') as txtfile:
    for row in pos_train_data:
        txtfile.write(row + '\n')
with open('data/pos_test.txt', 'w', encoding= 'utf-8') as txtfile:
    for row in pos_test_data:
        txtfile.write(row + '\n')

# define and train models on their training sets
text_model = fasttext.train_supervised(input= "data/text_train.txt", epoch= 25, lr= 0.5, wordNgrams= 2)
pos_model = fasttext.train_supervised(input= "data/pos_train.txt", epoch= 25, lr= 0.5, wordNgrams= 2)

# test models on their testing sets and display their precision (same value as recall for 2 label classification problems)
print(text_model.test("data/text_test.txt")[1])
print(pos_model.test("data/pos_test.txt")[1])

# save models
text_model.save_model('src/text_ai_classifier.bin')
pos_model.save_model('src/pos_combined_ai_classifier.bin')