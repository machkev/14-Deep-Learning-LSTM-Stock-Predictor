 # Summary of Deep Learning 
 ## Build, train and evaluate custom LSTM RNNs
 ## Overview
 
In this project we will use deep learning recurrent neural networks to model bitcoin closing prices. One model will use the FNG indicators to predict the closing price while the second model will use a window of closing prices to predict the nth closing price. 

The [Crypto Fear and Greed Index (FNG)](https://alternative.me/crypto/fear-and-greed-index/) analyzes emotions and sentiments from different sources to produce a daily FNG value for cryptocurrencies. 

During training we will experiment with different values for the following parameters: `window size(lookback window), number of input layers, number of epochs, and batch size`. Each model will be evaluated on unseen test data. This process can be repeated multiple times until we find a model with the best performace. We will vary the window size from 10 to 1 and then use the model to make predictions and compare them to actual values.

Each model will be built in separate notebooks. In order to make accurate comparisons between the two models, we need to maintain the same architecture and parameters during training and testing of each model. 

 ## Preprocessing the data for training and testing
 We set a random seed for reproducability.
"" 
from numpy.random import seed
seed(1)
from tensorflow import random
random.set_seed(2)
""

Ideally we should run multiple experiments to evaluate our model.

After importing and munging the data, we wukk yse the "window_data" function to generate X and y values for the model.  
We will then predict Closing Prices using a 10 day window of previous closing prices. Then, we wil experiment with window sizes anywhere from 1 to 10 and see how the model performance changes.

Each model will use 70% for training and 30% for testing. 
 
 In order to deal with possible outlier values and avoid skewed models, we need to apply the `MinMaxScaler` function `from sklearn.preprocessing` tools  to scale both x and y. This will arrange the feature and target values between 0 and 1, which will lead to more accurate predictions. 

 Note: It is good practice to fit the preprocessing function with the training dataset only. 
 We will convert the data by using the `numpy.reshape` function. 

 ## Build and train the LSTM RNN
 To begin designing a custom LSTM RNN we need to import the following modules: 
 ```python 
 from tensorflow.keras.models import Sequential 
 from tensorflow.keras.layers import Dense, LSTM, Dropout
 ```
 The `Sequential` model allows us to add and/or decrease stacks of LSTM layers. Each layer is able to identify and extract different patterns of time series sequences. The `Dropout` layer is a regularization method that reduces overfitting and improves performance by randomly dropping a fraction of the nodes created by the LSTM layer. The `Dense` layer "specifies the dimensionality of the output space". In this project the last layer flattens out the results from all the layers above and gives a single prediction. 

 After designing the model, we compile and fit it with our training dataset. It is here where we can further optimize our model by experimenting with the `fit` function parameters. The official [Keras guide](https://keras.io/guides/) provides an in-depth explanation of the LSTM RNN. 
 
 The snapshot below summarizes the architecture that resulted in the best performance for both the FNG and closing price models. 
 ```python
 window_size = 1 
 number_units = 30 # LSTM cells that learn to remember which historical patterns are important. 
 drop_out_fraction = 0.2 # 20% of neurons will be randomly dropped during each epoch
 model = Sequential()
# add 1st layer
model.add(LSTM(units = number_units, 
               return_sequences = True,  
               input_shape = (X_train.shape[1],1)
          ))
model.add(Dropout(drop_out_fraction))
# add 2nd layer
model.add(LSTM(units = number_units, return_sequences=True))
model.add(Dropout(drop_out_fraction))
# add 3rd layer 
model.add(LSTM(units = number_units))
model.add(Dropout(drop_out_fraction))
# add output layer
model.add(Dense(units=1))
# Compile the model
model.compile(loss='mean_squared_error', optimizer='adam', metrics=['mse'])
#Train the model using at least 10 epochs, no shuffling data, and using a batch size of 10
model.fit(X_train,y_train, epochs =10, batch_size=10, verbose = 1, shuffle=False)
 ```

## Evaluating the performance of each model
In this section we will use `X_test` and y_test data to evaluate the performance and use the X_test data to make predictions for each model. Then we will create a Dataframe to compare real prices vs predicted prices. We will then plot the real vs predicted values in a line chart.
We use the `scaler.inverse transform` function to recover the actual closing prices from the scaled values. 
'''
predicted_prices = scaler.inverse_transform(predictions)
real_prices = scaler.inverse_transform(y_test.reshape(-1, 1))
''''

Once the building, training, and testing proceedures have been successfully completed, we can see that the closing price model resulted in more accurate predictions. This is supported by the `mean square error loss` metric. The FNG model has a loss of .1283, and the closing price model has a lower loss of .0025. 

## Evaluation and performance comparison of the models

### Which model has a lower loss?

The LSTM Stock Predictor Using Closing Prices has a lower loss of 0.0156 compared to 0.1383 using the LSTM Stock Predictor Using Fear and Greed Index. In addition, using a smaller window size of 1 for the LSTM Stock Predictor Using Closing Prices has even a lower loss of 0.0049 compared with the larger window size 10.

### Which model tracks the actual values better over time?

The LSTM Stock Predictor Using Closing Prices tracks the actual values better over time. 


### Which window size works best for the model?

A smaller window size of 1 works best for the LSTM Stock Predictor Using Closing Prices as the model has a lower loss 
and tracks the actual values much closer. See chart below:

![Closing_Prices_window_1](/Starter_Code/closing_prices_window_1.png)![Closing_Prices_window_10](/Starter_Code/closing_prices_window_10.png)







