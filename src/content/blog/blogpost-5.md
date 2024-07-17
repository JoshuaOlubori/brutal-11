---
title: House Price Prediction 3
description: Implementing Deep Learning on scraped property images 
pubDate: 02/16/2024 01:00
author: Dennis Okwechime
tags: 
  - Deep Learning
  - Tensorflow

img: 'netherlands_houses.jpeg'
imgUrl: '../../assets/blog_covers/netherlands_houses.jpeg'
layout: ../../layouts/BlogPost.astro
category: Notebook
---

**Supercharging Predictions with Image Power!**

This approach takes things a step further by using powerful image recognition tools to extract even more features from property photos. These extra features can help the model make even more accurate rent price predictions.

Before we begin, we need to import the power horses that will enable our endeavours. Pre-trained image recognition models like MobileNet and InceptionV3, from TensorFlow's Keras library, are brought in as feature extractors. These models act like super-smart image analysis tools, having already learned a vast amount about identifying objects and patterns in pictures. Additionally, TensorFlow's building blocks like Dense layers and pooling functions are imported to create a custom model specifically designed to process property images and extract valuable features that can further enhance our rent price predictions.

## Importing modules and loading data

```python
import pandas as pd  # Importing pandas library and aliasing it as pd
import numpy as np  # Importing numpy library and aliasing it as np
from matplotlib import pyplot as plt  # Importing pyplot module from matplotlib library and aliasing it as plt
import os  # Importing os module for operating system dependent functionality
import seaborn as sns  # Importing seaborn library and aliasing it as sns
import warnings
warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)   # Setting pandas option to display all columns in DataFrame
plt.style.use('ggplot')  # Setting plot style to 'ggplot' from matplotlib

import tensorflow as tf
from tensorflow.keras.applications import MobileNet, InceptionV3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from sklearn.model_selection import train_test_split
from tqdm import tqdm
```

We load our data
```python
# Reading the CSV file containing all house prices
house_prices_df = pd.read_csv('All House Prices.csv'

# Reading the CSV file containing the list of processed images houses
processed_images_df = pd.read_csv('Processed Images Houses List.csv')
```
and then merge house prices data with processed images list based on 'id' and 'House ID'

```python
# Merging house prices data with processed images list based on 'id' and 'House ID'
final_df = pd.merge(house_prices_df, processed_images_df, left_on='id', right_on='House ID', how='right')

# Displaying the merged DataFrame
final_df.head()
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>Rent PCM</th>
      <th>House ID</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1125181</td>
      <td>2000.0</td>
      <td>1125181</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1225446</td>
      <td>3750.0</td>
      <td>1225446</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1956767</td>
      <td>3250.0</td>
      <td>1956767</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1948597</td>
      <td>950.0</td>
      <td>1948597</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003468</td>
      <td>2645.0</td>
      <td>2003468</td>
    </tr>
</table>


## Pre-processing Data and Extracting Features


Before feeding the property images to our deep learning models (MobileNet and InceptionV3), they need some prep work. This is where the `preprocess_image` function comes in. Here's what it does:

1.  **Load the Image:** It uses `load_img` from TensorFlow to grab the image from the "Combined Images" folder based on the property ID.
    
2.  **Resize It Right:** The image gets resized to 224x224 pixels, which is the specific size the MobileNet model expects.
    
3.  **From Picture to Numbers:** The function converts the image from a format that computers can understand for displaying pictures (likely JPEG or PNG) to a format they can use for calculations (NumPy array) using `img_to_array`.
    
4.  **MobileNet Magic:** Finally,  `preprocess_input` from MobileNet takes care of any special adjustments needed for the model to understand the image data properly (like normalization).
    

The code then loops through each property ID in the `final_df` DataFrame, following these steps to pre-process all the images. All the pre-processed images are stored in a list called `all_images`, which is then converted into a NumPy array named `X`.

Similarly, the rent prices per month (stored in the `'Rent PCM'` column of `final_df`) are extracted and stored as the target variable (y). Now, both the images (X) and the rent prices (y) are ready for the deep learning models to analyze and extract even more features to improve our rent price predictions!

```python
# Preprocessing Images

# Function to preprocess images
def preprocess_image(image_path):
    """
    Preprocesses an image for model input.
    """
    img = load_img(image_path, target_size=(224, 224))
    img_array = img_to_array(img)
    img_array = mobilenet.preprocess_input(img_array)
    return img_array

# List to store preprocessed images
all_images = []

# Preprocessing all images and appending to the list
for img_id in tqdm(final_df['id'].values):
    all_images.append(preprocess_image('Combined Images/' + str(img_id) + '.jpg'))

# Converting the list of images to numpy array
X = np.array(all_images)

# Extracting the target variable
y = final_df['Rent PCM'].values
```

## Data Splitting
The study divides the data (X and Y) into training and testing sets using the scikit-learn train_test_split function. The testing set will receive 20% of the data, with the remaining 80% going toward training, according to the test_size parameter, which is set to 0.20. To make the split consistent between runs, the random_state parameter is set to 42.

```python
# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```
## Evaluation metrics

To evaluate the deep learning models, the notebook uses common metrics (mean squared error, mean absolute error, R-squared) from `scikit-learn`. It also defines a custom `compute_metrics` function that takes a model name, predictions, and true values to calculate and print these metrics. A custom loss function was defined to optimize the model for rent price prediction. This can be used by calculating the root mean squared error between the predicted values (y_pred) and the true target values (y_true). This will be initialized during model training to minimize the root mean square error which is a known metric frequently employed in regression tasks.
```python
# Importing necessary module
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# Function to compute evaluation metrics
def compute_metrics(y_test, preds, model_name):
    """
    Compute evaluation metrics for a model and print the results.

    Parameters:
    y_test (array-like): True target values.
    preds (array-like): Predicted target values.
    model_name (str): Name of the model.

    """
    # Compute evaluation metrics
    rmse = mean_squared_error(y_test, preds, squared=False)
    print(f'Root Mean Squared Error: {rmse}')

    mae = mean_absolute_error(y_test, preds)
    print(f'Mean Absolute Error: {mae}')

    r_squared = r2_score(y_test, preds)
    print(f'R2 Score: {r_squared}')

# Custom metric function for root mean squared error
def root_mean_squared_error(y_true, y_pred):
    """
    Custom metric function for root mean squared error.
    """
    return tf.sqrt(tf.reduce_mean(tf.square(y_true - y_pred)))

```

## MobileNet Model

This section shows how to use the pre-trained MobileNet model to extract features from images for rent price prediction.

-   **Extracting Features:** The top layer of MobileNet is removed (include_top=False) to use it as a feature extractor.
    
-   **Building the Price Predictor:** A custom "head" is added with layers to process the extracted features and predict rent prices.
    
-   **Freezing the Base:** The pre-trained MobileNet weights are frozen (trainable=False) to focus training on the new head.
    
-   **Training and Evaluation:** The model is compiled with an Adam optimizer, custom metrics, and a learning rate of 0.01. It's then trained for 50 epochs on the training data (with validation on the testing data) using a batch size of 32.
```python
# Load MobileNet model without the top classification layer
base_model = MobileNet(weights='imagenet', include_top=False)

# Add custom regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1)(x)

# Combine base model with regression head
model = Model(inputs=base_model.input, outputs=predictions)
```

### Training the Regression Model with Early Stopping

```python
from keras.callbacks import EarlyStopping

# Freeze MobileNet layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=[root_mean_squared_error])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
```

### Making Predictions and Evaluating the Model

```python
preds = model.predict(X_test) # Making predictions on test data  

# Evaluating the model on the test set
compute_metrics(y_test, preds, 'mobile_net')
```

```python
# Creating DataFrame with predictions and actual values
predictions_df = pd.DataFrame({'mobile_net preds': preds.reshape(-1), 'actual': y_test})

# Displaying the DataFrame
display(predictions_df)

# Saving the predictions to a CSV file
predictions_df.to_csv('MobileNet Preds.csv', index=False)
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>mobile_net preds</th>
      <th>actual</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2576.895020</td>
      <td>1100.00</td>
    </tr>
    <tr>
      <th>1</th>
      <td>3124.343506</td>
      <td>3100.00</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2275.820312</td>
      <td>3450.00</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2407.327148</td>
      <td>3111.33</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1803.468262</td>
      <td>1700.00</td>
    </tr>
  </tbody>
</table>

> Table is trimmed to 5 rows

```python
# Save the model as a single file
tf.keras.models.save_model(model, 'mobilenet_model.h5')
```
## InceptionNet Model

Similar to MobileNet, the InceptionV3 section uses the pre-trained model (include_top=False) for feature extraction. It then adds a custom head for prediction, freezes the base model weights, and trains it for 50 epochs with a batch size of 32 (validation on test data). Model performance is evaluated using `compute_metrics` after training.

```python
# Load MobileNet model without the top classification layer
base_model = InceptionV3(weights='imagenet', include_top=False)

# Add custom regression head
x = base_model.output
x = GlobalAveragePooling2D()(x)
x = Dense(128, activation='relu')(x)
predictions = Dense(1)(x)

# Combine base model with regression head
model = Model(inputs=base_model.input, outputs=predictions)
```

### Training the Regression Model with Early Stopping

```python
from keras.callbacks import EarlyStopping

# Freeze MobileNet layers
for layer in base_model.layers:
    layer.trainable = False

# Compile model
model.compile(optimizer=Adam(learning_rate=0.01), loss='mean_squared_error', metrics=[root_mean_squared_error])

# Define early stopping criteria
early_stopping = EarlyStopping(monitor='val_loss', patience=5, verbose=1, restore_best_weights=True)

# Train the model with early stopping
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_data=(X_test, y_test), callbacks=[early_stopping])
```

>Epoch 1/50
43/43 [==============================] - 12s 170ms/step - loss: 3309032.5000 - root_mean_squared_error: 1757.0602 - val_loss: 2294656.0000 - val_root_mean_squared_error: 1435.0542
Epoch 2/50
43/43 [==============================] - 4s 94ms/step - loss: 2177483.5000 - root_mean_squared_error: 1444.5830 - val_loss: 2185812.7500 - val_root_mean_squared_error: 1421.3301
Epoch 3/50
43/43 [==============================] - 4s 94ms/step - loss: 2063109.2500 - root_mean_squared_error: 1410.3120 - val_loss: 2082349.2500 - val_root_mean_squared_error: 1385.8467
Epoch 4/50
43/43 [==============================] - 4s 98ms/step - loss: 1988515.5000 - root_mean_squared_error: 1374.6012 - val_loss: 1949107.7500 - val_root_mean_squared_error: 1327.6785
Epoch 5/50
43/43 [==============================] - 5s 108ms/step - loss: 1893848.8750 - root_mean_squared_error: 1345.3052 - val_loss: 1893979.8750 - val_root_mean_squared_error: 1300.5763
Epoch 6/50
43/43 [==============================] - 5s 107ms/step - loss: 1832199.1250 - root_mean_squared_error: 1325.9302 - val_loss: 1864974.0000 - val_root_mean_squared_error: 1289.7032
Epoch 7/50
43/43 [==============================] - 5s 112ms/step - loss: 1770368.8750 - root_mean_squared_error: 1311.3859 - val_loss: 1844105.1250 - val_root_mean_squared_error: 1290.2853
Epoch 8/50
43/43 [==============================] - 4s 104ms/step - loss: 1722729.0000 - root_mean_squared_error: 1292.4219 - val_loss: 1851787.1250 - val_root_mean_squared_error: 1300.7985
Epoch 9/50
43/43 [==============================] - 5s 108ms/step - loss: 1726474.6250 - root_mean_squared_error: 1289.3214 - val_loss: 1839029.7500 - val_root_mean_squared_error: 1293.8478
Epoch 10/50
43/43 [==============================] - 4s 99ms/step - loss: 1672335.0000 - root_mean_squared_error: 1274.1372 - val_loss: 1831734.5000 - val_root_mean_squared_error: 1277.8605
Epoch 11/50
43/43 [==============================] - 4s 95ms/step - loss: 1608407.5000 - root_mean_squared_error: 1252.2498 - val_loss: 1828215.5000 - val_root_mean_squared_error: 1284.6284
Epoch 12/50
43/43 [==============================] - 4s 104ms/step - loss: 1574184.6250 - root_mean_squared_error: 1240.9718 - val_loss: 1834282.3750 - val_root_mean_squared_error: 1288.3018
Epoch 13/50
43/43 [==============================] - 4s 96ms/step - loss: 1533516.6250 - root_mean_squared_error: 1220.4430 - val_loss: 1880856.1250 - val_root_mean_squared_error: 1315.7500
Epoch 14/50
43/43 [==============================] - 4s 91ms/step - loss: 1507111.7500 - root_mean_squared_error: 1204.7786 - val_loss: 1935800.3750 - val_root_mean_squared_error: 1341.5767
Epoch 15/50
43/43 [==============================] - 4s 104ms/step - loss: 1493655.6250 - root_mean_squared_error: 1199.6857 - val_loss: 1855977.1250 - val_root_mean_squared_error: 1297.1929
Epoch 16/50
43/43 [==============================] - ETA: 0s - loss: 1435834.8750 - root_mean_squared_error: 1182.0127Restoring model weights from the end of the best epoch: 11.
43/43 [==============================] - 4s 100ms/step - loss: 1435834.8750 - root_mean_squared_error: 1182.0127 - val_loss: 1878759.8750 - val_root_mean_squared_error: 1304.9607
Epoch 16: early stopping


### Making Predictions and Evaluating the Model
```python
preds = model.predict(X_test)  # Making predictions on test data

# Evaluating the model on the test set
compute_metrics(y_test, preds, 'Inception Net')
```
> 11/11 [==============================] - 3s 74ms/step 
> Root Mean Squared Error: 1352.11513186377 
> Mean Absolute Error: 940.5555313181322 
> R2 Score: 0.1721021259539165