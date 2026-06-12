import pandas as pd 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler,LabelEncoder, OneHotEncoder
import pickle

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.callbacks import EarlyStopping,TensorBoard
import datetime


#load the dataset
data=pd.read_csv("Housing.csv")
# print(data.head())


# label encode categorical variable needs 1d array so [] 

data['mainroad'] = data['mainroad'].map({'yes': 1, 'no': 0})
data['guestroom'] = data['guestroom'].map({'yes': 1, 'no': 0})
data['basement'] = data['basement'].map({'yes': 1, 'no': 0})
data['hotwaterheating'] = data['hotwaterheating'].map({'yes': 1, 'no': 0})
data['airconditioning'] = data['airconditioning'].map({'yes': 1, 'no': 0})
data['prefarea'] = data['prefarea'].map({'yes': 1, 'no': 0})
# pd.set_option('display.max_columns', None)
# print(data.head())


# one hot encodeing geography needs 2d array so [[]]
onehot_encoder_fur=OneHotEncoder(sparse_output=False)
fur_encoder=onehot_encoder_fur.fit_transform(data[['furnishingstatus']])
fur_encoded_df=pd.DataFrame(
    fur_encoder,
    columns=onehot_encoder_fur.get_feature_names_out(['furnishingstatus'])
)
# print(fur_encoded_df)


# combine all the columns with the original data
data=pd.concat([data.drop("furnishingstatus",axis=1),fur_encoded_df],axis=1)
# print(data)



with open('onehot_encoder_fur.pkl', 'wb') as file:
    pickle.dump(onehot_encoder_fur, file)


# divide the dataset into independent and dependent features
x = data.drop('price', axis=1)
y = data['price']

# split the data
x_train,x_test,y_train,y_test = train_test_split(
    x,y,test_size=0.2,random_state=42
)


# Scale target variable (price)
y_scaler = StandardScaler()

y_train = y_scaler.fit_transform(
    y_train.values.reshape(-1, 1)
)

y_test = y_scaler.transform(
    y_test.values.reshape(-1, 1)
)

with open('y_scaler.pkl', 'wb') as file:
    pickle.dump(y_scaler, file)

# scale
scaler = StandardScaler()

x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)

    

with open('scaler.pkl', 'wb') as file:
    pickle.dump(scaler, file)
    
    
model = Sequential([
    Dense(64, activation="relu", input_shape=(x_train.shape[1],)),
    Dense(32, activation="relu"),
    Dense(1)
])
print(model.summary())
# print(data['price'].describe())



model.compile(
    optimizer="adam",
    loss="mean_absolute_error",
    metrics=["mae"]
)


log_dir = "regressionlogs/fit/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

tensorflow_callback = TensorBoard(
    log_dir=log_dir,
    histogram_freq=1
)

early_stopping_callback=EarlyStopping(monitor="val_loss",patience=10,restore_best_weights=True)


history=model.fit(
    x_train,y_train,validation_data=(x_test,y_test),epochs=100,
    callbacks=[tensorflow_callback,early_stopping_callback]
)


test_loss, test_mae = model.evaluate(x_test, y_test)

print(f"Test Loss: {test_loss}")
print(f"Test MAE: {test_mae}")

model.save('house_model.h5')