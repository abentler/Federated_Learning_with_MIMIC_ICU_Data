import os
import random
import time
import numpy as np
import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from keras.utils import to_categorical

import openfl.native as fx
from openfl.federated import FederatedModel,FederatedDataSet

# ensure proper location of data files
icu_stays_fn = "data/icu/icustays.csv/icustays.csv"
input_events_fn = "data/icu/inputevents.csv/inputevents.csv"
output_events_fn = "dataicu/outputevents.csv/outputevents.csv"
ingred_events_fn = "data/icu/ingredientevents.csv/ingredientevents.csv"
proced_events_fn = "data/icu/procedureevents.csv/procedureevents.csv"
chart_events_fn = "data/icu/chartevents.csv/chartevents.csv"
date_events_fn = "data/icu/datetimeevents.csv/datetimeevents.csv"

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

# min-max scaling normalization
event_counts[item_columns] = event_counts[item_columns] / event_counts[item_columns].max()
event_counts['los'] = event_counts['los'].astype(int) # continuous: / max_stay

#print(event_counts[:3])
print("DONE\n")

print("Creating training, validation, and testing datasets...")
event_counts = event_counts.sample(frac=1, ignore_index=True) # shuffle the data

# divide into training, testing, validation datasets
cutoff_training = (int) (len(event_counts) * 0.8)
cutoff_validation = cutoff_training + (int) (len(event_counts) * 0.1)
training = event_counts[:cutoff_training]
validation = event_counts[cutoff_training:cutoff_validation]
testing = event_counts[cutoff_validation:]

print("DONE\n")

print("Training model...")

# separate training, validation, and testing inputs and outputs
# convert to numpy for keras compatability
train_x = training[item_columns].to_numpy()
train_y = training['los'].to_numpy()
valid_x = validation[item_columns].to_numpy()
valid_y = validation['los'].to_numpy()
test_x = testing[item_columns].to_numpy()
test_y = testing['los'].to_numpy()

# print(training[:3]['los'])
# print(train_y[:3])

# print(training[:3][item_columns])
# print(train_x[:3])

# getting length and width values for future use
number_of_classes = max(event_counts['los']) + 1
number_of_features = len(item_columns)

# convert data to format accepted by keras model
train_y = to_categorical(train_y, num_classes=number_of_classes)
valid_y = to_categorical(valid_y, num_classes=number_of_classes)

# model setup, parameters
model = Sequential()
model.add(Dense(number_of_classes,activation='softmax',input_dim=number_of_features))
model.compile(optimizer='adam', loss='categorical_crossentropy')
# training the model
model.fit(train_x, train_y, 
          validation_data=(valid_x, valid_y), epochs=150)


print("Testing model...")

predictions = np.argmax(model.predict(test_x), axis=1)
print(predictions)

error = 0.0
accuracy = 0.0
test_matrix = np.zeros((number_of_classes, number_of_classes), dtype=float)

for i in range(0, len(predictions)):
   this_error = (predictions[i] - test_y[i])
   #print(testing.iloc[i]['los'], predictions[i], this_error)
   error += abs(predictions[i] - test_y[i])
   if(predictions[i] == test_y[i]):
      accuracy += 1
   test_matrix[predictions[i]][test_y[i]] += 1


print("DONE\n")
print(test_matrix)

p = [0] * number_of_classes
tp = [0] * number_of_classes
r = [0] * number_of_classes

for days in range(0, number_of_classes):
   total = sum(test_matrix[days])
   tp[days] = test_matrix[days][days]
   if total > 0:
      p[days] = tp[days] / total
   for i in range(0, number_of_classes):
      r[days] += test_matrix[i][days]

for i in range(0, number_of_classes):
   if(r[i] > 0):
      r[i] = tp[i] / r[i]

print("PRECISION VALUES: ", p)
print("RECALL VALUES: ", r)

error = error / len(predictions)
accuracy = accuracy / len(predictions)
print("AVG ERROR:", error)
print("ACCURACY: ", accuracy)
