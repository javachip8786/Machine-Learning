#GROUP 49
#Arham Ali
#Ayush Arpit
#Siddharth Singh (Group Leader)

# In[134]:


#libraries required
import numpy as np  #for fast mathematical computations and handling vectorizations
import pandas as pd # for handling raw data read from file. 
from sklearn.preprocessing import StandardScaler    # Scale data to Standard dist
from sklearn.linear_model import LogisticRegression # Logistic Regression implementation from sci-kit learn 
from keras.models import Sequential # Model used for stacking layers of neurons to generate a neural network
from keras.layers import Dense # a layer of neurons in the neural network
import random # Python's random library to generate random numbers


# In[2]:

print('Data Preprocessing...')
#importing training data and test data CSVs into the program using the pandas read_csv fn
train_data = pd.read_csv('input/train_data.csv')
test_data = pd.read_csv('input/test_data.csv')


# In[3]:


#converting training set into file with only sequences
train_data.Sequence.to_csv('input/sequence_data.csv', index= False)


# In[4]:


#converting train data to amino acid features using Pfeatures and then importing it
sequence_data = pd.read_csv('input/final_amino_acid_result.csv')


# In[5]:


#importing library for oversampling
from imblearn.combine import SMOTETomek


# In[6]:


x_train = sequence_data.values  # taking out values from dataframe
x_train = x_train[:, 1:]    #dropping sequence number
x_train = x_train.astype(np.float64) # converting from object type to float


# In[7]:

#using label as out attribute to predict
y_train = train_data['label'].values


# In[8]:

'''
oversampling data to have equal number of training samples
 and test samples since the provided training dataset 
 was unbalanced. 1801 interacting and 35090 non interacting samples.
 fit_resample method of SMOTETomek returns equal number of data points for both labels
'''
smk = SMOTETomek(random_state=41)
x_res, y_res = smk.fit_resample(x_train, y_train) 


# In[9]:


# shape of data after oversampling
print(x_res.shape, y_res.shape)


# In[11]:


#exporting sequence data to be used for conversion by pfeature
test_data.Sequence.to_csv('input/test_sequence.csv', index=False, header=None)


# In[10]:

#converting testing sequence to amino acid using Pfeatures and then importing it
x_test = pd.read_csv('input/final_test_result.csv').values[:, 1:]



# In[21]:

#importing sample dataset (submission format)
sample_data = pd.read_csv('input/sample.csv')
#Data Preprocessing complete! Begin with testing out models
# # Logistic Regression Model

# In[13]:


print('Logistic Regression Model...')
# training using Logistic Regression Model
clf = LogisticRegression(random_state=0)
clf.fit(x_res, y_res) # fitting the oversampled data into the LR model


# In[14]:


# Testing Model
preds = clf.predict(x_test) #generating predictions against the test data


# In[15]:

#counting 1's in preds
print("Count of interacting samples in LR Model : ", preds.tolist().count(1))


# In[40]:


#importing sample data
sample_data = pd.read_csv('input/sample.csv')   #importing submission file 
sample_data.Label  = preds # assigning Label column to generated predictions
sample_data.to_csv('output/lr_output.csv', index=False) # exporting the file to be submitted
print('Predictions stored at output/lr_output.csv')

print('Neural Network Model...')
# # Deep learning Keras Ensemble
# 

# In[136]:


import tensorflow as tf #the tensorflow library which includes deeplearning functions


# In[130]:

#normalizing the data set
sc = StandardScaler()   # initializing the scaling function
pfeature_data = np.column_stack((sc.fit_transform(x_train), y_train))   # stacking columns inorder to get an array with both x's and y's. This repreesents the training data
pdata = pd.DataFrame(pfeature_data) #converting into dataframe object to run comparison queries and segragating 0 and 1 values
ones_data = (pdata[pdata[20] == 1].values)  #extracting interacting sequences (with label 1) from the defined dataframe object
zeros_data = (pdata[pdata[20] == 0].values) #extracting non-interacting sequence (label 0 ) from the defined dataframe object
ones_data = ones_data[:, :-1].astype(np.float32) #taking input values of interacting seq and converting dtype from object to float
zeros_data = zeros_data[:, :-1].astype(np.float32) #taking input values of non-interacting seq and converting dtype from object to float


# In[137]:

#Ensembling models and generating 11 predictions
out_preds = [] # array which stores all the 11 model predictions
for i in range(11):
    model = Sequential() # initializing a neural network model
    model.add(Dense(16,input_dim= ones_data.shape[1], activation='relu')) #adding a input Dense layer of 16 neurons
    model.add(Dense(4, activation='relu')) # adding a hidden Dense layer of 4 neurons
    model.add(Dense(1, activation='sigmoid'))# adding a output Dense layer of 1 neuron
    # taking a random number to choose 1801 random samples from zeros_data for training
    i = random.randint(0, zeros_data.shape[0] - ones_data.shape[0]-1) 
    sample_zero = zeros_data[i: i+ ones_data.shape[0]] #taking 1801 random samples from zeros_data
    x_data = np.row_stack((sample_zero, ones_data)) # generating the sample training input data
    # generating the sample training output data:
    y_data = np.row_stack((np.zeros(ones_data.shape[0]), np.ones(ones_data.shape[0]))).reshape(-1,)
    #defining early_stopping to stop the training when change (delta) becomes insignificant. Also maintaining the best weights
    early_stopping = tf.keras.callbacks.EarlyStopping(
    patience=10,
    restore_best_weights=True,
    min_delta=0.0005,
    )
    # defining loss, optimizer and metrics paramaeters. Compiling the model:
    model.compile(loss = 'binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    # fitting the model with the given input data:
    history = model.fit(sc.fit_transform(x_data), y_data, epochs= 256, verbose=1,callbacks=[early_stopping])
    # appending the current model predictions to an output array
    out_preds.append(model.predict(sc.fit_transform(x_test)))


# In[152]:


# final_preds = np.array(out_preds).round().sum(axis=0)
#generating final predictions by reshaping to required size and summing all predictions
final_preds = np.array(out_preds).reshape((11,9582,)).sum(axis = 0)
#applying majority voting in order to obtain final predicted values:
for i in range(len(final_preds)):# putting values in array according to threshold
    if final_preds[i] > 5: #when more than half of the models predict it as interacting
        final_preds[i] = 1
    else:   # when more than half of the models predict it as non interacting
        final_preds[i] = 0
final_preds.tolist().count(1)#counting no of 1's in our model


# In[156]:


sample_data.Label = final_preds.astype(np.int64) # converting from floating point data to integers
sample_data.to_csv('output/NN_ensemble.csv', index=False)#csv file output of NN_ensemble
print('Predictions stored at output/NN_ensemble.csv')

# In[ ]:

#converting sequence to ASCII value
def seq_to_ascii(x):
    o = [] #initialize an output array storing ascii vectors of all sequences
    for i in range(x.shape[0]):
        out = []    # ascii vector for a sequence
        for c in x[i]:
            #ASCII conversion of every sequence character
            #converting every sequence char into a integer representing its corresponding ASCII value. Appending its relative location from 'A' to ascii vector
            out.append(ord(c) - ord('A')) 
        o.append(out)   #adding the final ascii vector into the o array
    return np.array(o) #convert and return the numpy array of o vector


# 
print('Categorical Naive Bayes...')
# # Categorical Naive bayes with ensemble Learning
# 

# In[122]:

#converting the sequence to ascii values
ones_data = seq_to_ascii(train_data[train_data['label'] == 1]['Sequence'].values)#converting where label=1
zeros_data = seq_to_ascii(train_data[train_data['label'] == 0]['Sequence'].values)#converting where label=0
test_x = seq_to_ascii(test_data.Sequence.values) # generating ascii features for every sequence


# In[123]:


ones_data.shape


# In[124]:


x_data = np.row_stack((zeros_data[1: 1 + ones_data.shape[0]], ones_data))# generating the sample training input data
# generating the sample training output data:
y_data = np.row_stack((np.zeros(ones_data.shape[0]), np.ones(ones_data.shape[0]))).reshape(-1,)


# In[125]:


from sklearn.naive_bayes import CategoricalNB   # sklearn implementation for Categorical Naive Bayes
out_preds = [] # represents a vector to store predictions of each of the 11 models
for i in range(11):
    model = CategoricalNB()#applying the naive bayes model
    i = random.randint(0, zeros_data.shape[0] - ones_data.shape[0]-1)
    sample_zero = zeros_data[i: i+ ones_data.shape[0]]
    x_data = np.row_stack((sample_zero, ones_data))# generating the sample training input data
    # generating the sample training output data:
    y_data = np.row_stack((np.zeros(ones_data.shape[0]), np.ones(ones_data.shape[0]))).reshape(-1,)
    model.fit(x_data, y_data)#fitting given sample data into the NB model
    out_preds.append(model.predict(test_x))#generating predictions and appending to out_preds


# In[126]:

# adding all predictions across the rows to get a sum of predictions for every test data
final_preds = np.array(out_preds).sum(axis=0) 
for i in range(len(final_preds)): #Majority voting
    if final_preds[i] > 5: # if more than half models predicts interacting then assign label 1
        final_preds[i] = 1
    else: # else assign label 0
        final_preds[i] = 0 
final_preds.tolist().count(1)


# In[127]:


sample_data.Label = final_preds.astype(np.int64) # convverting from float to input
sample_data.to_csv('output/CategoricalNaiveBayesEnsemble.csv', index=False)#final output as csv file
print('Predictions stored at output/CategoricalNaiveBayesEnsemble.csv')
