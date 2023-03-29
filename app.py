import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
np.set_printoptions(suppress=True)
from nsepy import get_history
from datetime import datetime

startDate=datetime(2019, 1,1)
endDate=datetime(2020, 10, 5)
StockData=get_history(symbol='INFY', start=startDate, end=endDate)
# print(StockData.shape)
# print(StockData.head())



StockData['TradeDate']=StockData.index
StockData.plot(x='TradeDate', y='Close', kind='line', figsize=(20,6), rot=20)
# plt.show()



FullData=StockData[['Close']].values
# print(FullData[0:5])
from sklearn.preprocessing import StandardScaler, MinMaxScaler
sc=MinMaxScaler()
DataScaler = sc.fit(FullData)
X=DataScaler.transform(FullData)
# print('### After Normalization ###')
# print(X[0:5])



X_samples = list()
y_samples = list()
NumerOfRows = len(X)
TimeSteps=10  
for i in range(TimeSteps , NumerOfRows , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i]
    X_samples.append(x_sample)
    y_samples.append(y_sample)
X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
# print('\n#### Input Data shape ####')
# print(X_data.shape)
y_data=np.array(y_samples)
y_data=y_data.reshape(y_data.shape[0], 1)
# print('\n#### Output Data shape ####')
# print(y_data.shape)



TestingRecords=5
X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]
# print('\n#### Training Data shape ####')
# print(X_train.shape)
# print(y_train.shape)
# print('\n#### Testing Data shape ####')
# print(X_test.shape)
# print(y_test.shape)



# for inp, out in zip(X_train[0:2], y_train[0:2]):
#     print(inp,'--', out)




TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
# print("Number of TimeSteps:", TimeSteps)
# print("Number of Features:", TotalFeatures)




from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
regressor = Sequential()
regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
regressor.add(Dense(units = 1))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
import time
StartTime=time.time()
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
EndTime=time.time()
print("## Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes ##')




predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
orig=y_test
orig=DataScaler.inverse_transform(y_test)
print('Accuracy:', 100 - (100*(abs(orig-predicted_Price)/orig)).mean())
import matplotlib.pyplot as plt
plt.plot(predicted_Price, color = 'blue', label = 'Predicted Volume')
plt.plot(orig, color = 'lightblue', label = 'Original Volume')
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.xticks(range(TestingRecords), StockData.tail(TestingRecords)['TradeDate'])
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(6)
plt.show()


'''


TrainPredictions=DataScaler.inverse_transform(regressor.predict(X_train))
TestPredictions=DataScaler.inverse_transform(regressor.predict(X_test))
FullDataPredictions=np.append(TrainPredictions, TestPredictions)
FullDataOrig=FullData[TimeSteps:]
plt.plot(FullDataPredictions, color = 'blue', label = 'Predicted Price')
plt.plot(FullDataOrig , color = 'lightblue', label = 'Original Price')
plt.title('Stock Price Predictions')
plt.xlabel('Trading Date')
plt.ylabel('Stock Price')
plt.legend()
fig=plt.gcf()
fig.set_figwidth(20)
fig.set_figheight(8)
plt.show()





Last10Days=np.array([1002.15, 1009.9, 1007.5, 1019.75, 975.4,
            1011.45, 1010.4, 1009,1008.25, 1017.65])
Last10Days=DataScaler.transform(Last10Days.reshape(-1,1))
NumSamples=1
TimeSteps=10
NumFeatures=1
Last10Days=Last10Days.reshape(NumSamples,TimeSteps,NumFeatures)
predicted_Price = regressor.predict(Last10Days)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
print(predicted_Price)




print('Original Prices')
print(FullData[-10:])
print('###################')
X=X.reshape(X.shape[0],)
print('Scaled Prices')
print(X[-10:])




X_samples = list()
y_samples = list()
NumerOfRows = len(X)
TimeSteps=10  
FutureTimeSteps=5 
for i in range(TimeSteps , NumerOfRows-FutureTimeSteps , 1):
    x_sample = X[i-TimeSteps:i]
    y_sample = X[i:i+FutureTimeSteps]
    X_samples.append(x_sample)
    y_samples.append(y_sample)
X_data=np.array(X_samples)
X_data=X_data.reshape(X_data.shape[0],X_data.shape[1], 1)
print('### Input Data Shape ###') 
print(X_data.shape)
y_data=np.array(y_samples)
print('### Output Data Shape ###') 
print(y_data.shape)




TestingRecords=5
X_train=X_data[:-TestingRecords]
X_test=X_data[-TestingRecords:]
y_train=y_data[:-TestingRecords]
y_test=y_data[-TestingRecords:]
print('\n#### Training Data shape ####')
print(X_train.shape)
print(y_train.shape)
print('\n#### Testing Data shape ####')
print(X_test.shape)
print(y_test.shape)




for inp, out in zip(X_train[0:2], y_train[0:2]):
    print(inp)
    print('====>')
    print(out)
    print('#'*20)





TimeSteps=X_train.shape[1]
TotalFeatures=X_train.shape[2]
print("Number of TimeSteps:", TimeSteps)
print("Number of Features:", TotalFeatures)





from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
regressor = Sequential()
regressor.add(LSTM(units = 10, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
regressor.add(LSTM(units = 5, activation = 'relu', input_shape = (TimeSteps, TotalFeatures), return_sequences=True))
regressor.add(LSTM(units = 5, activation = 'relu', return_sequences=False ))
regressor.add(Dense(units = FutureTimeSteps))
regressor.compile(optimizer = 'adam', loss = 'mean_squared_error')
import time
StartTime=time.time()
regressor.fit(X_train, y_train, batch_size = 5, epochs = 100)
EndTime=time.time()
print("############### Total Time Taken: ", round((EndTime-StartTime)/60), 'Minutes #############')





predicted_Price = regressor.predict(X_test)
predicted_Price = DataScaler.inverse_transform(predicted_Price)
print('#### Predicted Prices ####')
print(predicted_Price)
orig=y_test
orig=DataScaler.inverse_transform(y_test)
print('\n#### Original Prices ####')
print(orig)




import matplotlib.pyplot as plt

for i in range(len(orig)):
    Prediction=predicted_Price[i]
    Original=orig[i]
    
    # Visualising the results
    plt.plot(Prediction, color = 'blue', label = 'Predicted Volume')
    plt.plot(Original, color = 'lightblue', label = 'Original Volume')

    plt.title('### Accuracy of the predictions:'+ str(100 - (100*(abs(Original-Prediction)/Original)).mean().round(2))+'% ###')
    plt.xlabel('Trading Date')
    
    startDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1)
    endDateIndex=(FutureTimeSteps*TestingRecords)-FutureTimeSteps*(i+1) + FutureTimeSteps
    TotalRows=StockData.shape[0]

    plt.xticks(range(FutureTimeSteps), StockData.iloc[TotalRows-endDateIndex : TotalRows-(startDateIndex) , :]['TradeDate'])
    plt.ylabel('Stock Price')

    plt.legend()
    fig=plt.gcf()
    fig.set_figwidth(20)
    fig.set_figheight(3)
    plt.show()




import numpy as np
Last10DaysPrices=np.array([1376.2, 1371.75,1387.15,1370.5 ,1344.95, 
                   1312.05, 1316.65, 1339.45, 1339.7 ,1340.85])
Last10DaysPrices=Last10DaysPrices.reshape(-1, 1)
X_test=DataScaler.transform(Last10DaysPrices)
NumberofSamples=1
TimeSteps=X_test.shape[0]
NumberofFeatures=X_test.shape[1]
X_test=X_test.reshape(NumberofSamples,TimeSteps,NumberofFeatures)
Next5DaysPrice = regressor.predict(X_test)
Next5DaysPrice = DataScaler.inverse_transform(Next5DaysPrice)
print(Next5DaysPrice)



'''
