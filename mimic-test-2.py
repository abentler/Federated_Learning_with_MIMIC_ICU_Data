import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset
import pandas as pd
from sklearn import linear_model

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical

import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet

fx.init()

def build_model(num_features,num_classes): # features is tuple (#,)
   model = Sequential()
   model.add(Dense(num_classes,activation='softmax',input_dim=num_features[0]))
   model.compile(optimizer='adam', loss='categorical_crossentropy')
   return model

# ensure proper location of data files
icu_stays_fn = "data/MIMIC-IV/icu/icustays.csv/icustays.csv"
input_events_fn = "data/MIMIC-IV/icu/inputevents.csv/inputevents.csv"
output_events_fn = "data/MIMIC-IV/icu/outputevents.csv/outputevents.csv"
ingred_events_fn = "data/MIMIC-IV/icu/ingredientevents.csv/ingredientevents.csv"
proced_events_fn = "data/MIMIC-IV/icu/procedureevents.csv/procedureevents.csv"
chart_events_fn = "data/MIMIC-IV/icu/chartevents.csv/chartevents.csv"
date_events_fn = "data/MIMIC-IV/icu/datetimeevents.csv/datetimeevents.csv"

print("Reading data from CSV...")
icu_stays = pd.read_csv(icu_stays_fn, usecols=['stay_id', 'los'])
input_events = pd.read_csv(input_events_fn, usecols=['stay_id', 'itemid'])
output_events = pd.read_csv(output_events_fn, usecols=['stay_id', 'itemid'])
ingred_events = pd.read_csv(ingred_events_fn, usecols=['stay_id', 'itemid'])
proced_events = pd.read_csv(proced_events_fn, usecols=['stay_id', 'itemid'])
chart_events = pd.read_csv(chart_events_fn, usecols=['stay_id', 'itemid'])
date_events = pd.read_csv(date_events_fn, usecols=['stay_id', 'itemid'])
print("DONE\n")

# merge data to map icu events to length of stay information for each patient
print("Merging data...")
icu_data = pd.merge(input_events, icu_stays, on="stay_id", how="inner")
icu_data2 = pd.merge(output_events, icu_stays, on="stay_id", how="inner")
icu_data3 = pd.merge(ingred_events, icu_stays, on="stay_id", how="inner")
icu_data4 = pd.merge(proced_events, icu_stays, on="stay_id", how="inner")
icu_data5 = pd.merge(chart_events, icu_stays, on="stay_id", how="inner")
icu_data6 = pd.merge(date_events, icu_stays, on="stay_id", how="inner")
icu_data = pd.concat([icu_data, icu_data2, icu_data3, icu_data4, icu_data5, icu_data6], 
                     ignore_index=True)
#print(icu_data[:3])
event_counts = icu_data.groupby(['stay_id', 'los', 'itemid']).size().unstack(fill_value=0)

# Reset the index to make stay_id a column again
event_counts = event_counts.reset_index()
item_columns = event_counts.columns[2:] # get values for all item columns
print("DONE\n")

print("Normalizing data...")
max_stay = event_counts['los'].max()

# might need to scale normalization between -1 and 1 per sklearn
event_counts[item_columns] = event_counts[item_columns] / event_counts[item_columns].max()
event_counts['los'] = event_counts['los'].astype(int) # continuous: / max_stay

#print(event_counts[:3])
print("DONE\n")

print("Creating training and testing datasets...")
event_counts = event_counts.sample(frac=1, ignore_index=True) # shuffle the data

number_of_classes = max(event_counts['los']) + 1
number_of_features = len(item_columns)

cutoff_training = (int) (len(event_counts) * 0.8)
cutoff_validation = cutoff_training + (int) (len(event_counts) * 0.1)
training = event_counts[:cutoff_training]
validation = event_counts[cutoff_training:cutoff_validation]
testing = event_counts[cutoff_validation:]

train_x = training[item_columns].to_numpy()
train_y = training['los'].to_numpy()
valid_x = validation[item_columns].to_numpy()
valid_y = validation['los'].to_numpy()
test_x = testing[item_columns].to_numpy()
test_y = testing['los'].to_numpy()

#train_y = to_categorical(train_y, num_classes=number_of_classes)
#valid_y = to_categorical(valid_y, num_classes=number_of_classes)

print("DONE\n")

print("Training model...")
linreg = linear_model.LogisticRegression(max_iter=200)
linreg.fit(training[item_columns], training['los'])

fl_data = FederatedDataSet(train_x,train_y,test_x,test_y,
                           batch_size=32, num_classes=number_of_classes) #,num_classes=classes)
fl_model = FederatedModel(build_model, data_loader=fl_data)

collaborator_models = fl_model.setup(num_collaborators=5)
collaborators = {'one':collaborator_models[0],'two':collaborator_models[1]}

final_fl_model = fx.run_experiment(collaborators, 
                                   override_config={"aggregator.settings.rounds_to_train": 5})


#print("\nTRAINING")
#print(training[:3])
#print("\nTESTING")
#print(testing[:3][item_columns])

print("DONE\n")

print("Testing model...")
predictions = linreg.predict(testing[item_columns])
error = 0.0
accuracy = 0.0

for i in range(0, len(predictions)):
   this_error = (predictions[i] - testing.iloc[i]['los'])
   #print(testing.iloc[i]['los'], predictions[i], this_error)
   error += abs(predictions[i] - testing.iloc[i]['los'])
   if(predictions[i] == testing.iloc[i]['los']):
      accuracy += 1

print("DONE\n")

error = error / len(predictions)
accuracy = accuracy / len(predictions)
print("AVG ERROR:", error)
print("ACCURACY: ", accuracy)
print(linreg.n_iter_)