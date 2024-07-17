---
title: House Price Prediction I 
description: First part of my masters thesis -- House Rent Prediction using Machine Learning and Deep Learning Algorithms, The case of the United Kingdom
pubDate: 02/15/2024 02:55
author: Dennis Okwechime
tags: 
  - Web scraping
  - Selenium

img: 'netherlands_houses.jpeg'
imgUrl: '../../assets/blog_covers/netherlands_houses.jpeg'
layout: ../../layouts/BlogPost.astro
category: Notebook
---


### Introduction

The recent fluctuation in the U.K. housing market highlights the substantial impact of home price changes on consumer spending, financial markets, and the overall macroeconomy (Bloomberg, 2023). Developing a reliable forecasting model could offer valuable insights to central banks, financial regulatory organizations, and other economic entities. A precise forecast of house prices is crucial for potential homeowners, developers, investors, appraisers, tax assessors, mortgage lenders, insurers, and other participants in the real estate market, as stated by Rafiei and Adeli (2016).
The aim of this project is to:

- To scrape and gather extensive property information from leading real estate websites, ensuring a wide range of property kinds, locations, and attributes. 
- To develop a predictive analytics model using Linear Regression, Random Forest, XGBoost, and Catboost to analyze a dataset of housing prices.
- To determine the effectiveness of the created predictive model in comparison to previous research. 
- To choose the top two models for creating an ensemble model to improve the reliability of predicting housing prices using the machine learning and deep learning approach(CNN). 
- To implement the ensemble models for testing purposes within the mobile web environment.


---

Data was collected on 3565 houses in 5 cities, namely, London, Birmingham, Liverpool, Glasgow & Manchester.

Each listing includes:
-   Text description
-   Photos of the house exterior, bedrooms, kitchen & bathroom
-   Location postcode
-   Number of bedrooms, bathrooms & gardens ️

Data comes from open rent websites & is manually verified using real estate info.



## House Data Scraping

A script was developed  to scrape the data. These were the core technologies used.
-   **Selenium:** Automates tasks within a web browser, allowing the script to navigate and interact with Open Rent.
-   **Beautiful Soup:** Parses the HTML content of webpages, helping the script extract specific details about each property.
-   **Pandas:** Go-to library for manipulating tabular data
-   **google_colab_selenium:** Enables the script to run Selenium within the Google Colab environment.

### **Step 1: Importing the Essentials**

The script starts by importing the necessary libraries. These include:
-   **time:** Manages delays between actions, ensuring the script doesn't overwhelm the website.
-   **Pandas:** As mentioned above, for data manipulation and storage.
-   **Beautiful Soup:** For parsing HTML content.
-   **Selenium:** To control the web browser.
-   **tqdm** Creates progress bars to visualize the scraping process.
-   **requests** used for making HTTP requests
-   **Beautiful Soup (bs4):** Parses the HTML content of openrent.co.uk

```python
# Import necessary libraries
import time # For adding delays in the script
from PIL import Image # For working with images
from selenium import webdriver # For controlling the web browser
from selenium.webdriver.chrome.options import Options # For configuring Chrome options
from selenium.webdriver.common.by import By # For locating elements on a webpage
from tqdm import tqdm # For creating progress bars
import requests # For making HTTP requests
import pandas as pd # For working with data in tabular format
from bs4 import BeautifulSoup # For parsing HTML content
pd.set_option('display.max_columns', None) # To display all columns in pandas DataFrame
```

### **Step 2: Configuring the Chrome Browser**

The script then configures the Chrome browser specifically for web scraping. This involves:

-   Setting the desired window size for efficient data extraction.
-   Disabling unnecessary pop-ups and info bars to streamline the process.
-   Ignoring certificate errors (if encountered) to ensure smooth operation.
-   Launching Chrome in incognito mode for better privacy and resource management.

These configurations are applied to a Chrome WebDriver instance created using the google_colab_selenium library.

```python
# Import the google_colab_selenium module as gs
import google_colab_selenium as gs
# Create Chrome options object
options = Options()
# Add extra options to the Chrome browser
options.add_argument("--window-size=1920,1080")  # Set the window size
options.add_argument("--disable-infobars")  
# Disable the infobars
options.add_argument("--disable-popup-blocking")  # Disable pop-ups
options.add_argument("--ignore-certificate-errors")  # Ignore certificate errors
options.add_argument("--incognito")  # Use Chrome in incognito mode
# Create a Chrome WebDriver instance with the specified options
driver = gs.Chrome(options=options)
```

###  Step 3: City Selection, URL Construction, and Page Loading

The user is granted the flexibility to specify the city for property search by modifying the CITY variable within the script. Due to the selected city, the script dynamically develops a relevant URL of OpenRent, allowing it to search for properties within the selected area. Next, the script navigates to the constructed URL and uses a very intelligent scroll mechanism to load all available property listings on the page. The number of scrolls is controlled by the NO_OF_SCROLLS variable; this number gets repeatedly taken down the page, whereby the scrolling pause time for every attempt is controlled by the SCROLL_PAUSE_TIME variable (See [listing 1.3](#a)),
allowing the page to load seamlessly.

 <a name="a"></a>
```python
# Define the city for property search
#CITY = 'birmingham-west-midlands'
CITY =   'glasgow-city' #'manchester' #'liverpool-merseyside'
# Define scroll parameters
SCROLL_PAUSE_TIME = 0.5   # Time to pause between each scroll
NO_OF_SCROLLS = 30   # Number of times to scroll
# Construct the URL for property search in the specified city
url = f'https://www.openrent.co.uk/properties-to-rent/{CITY}'
# Navigate to the constructed URL
driver.get(url)
# Get the initial scroll height of the webpage
last_height = driver.execute_script("return document.body.scrollHeight")

# Loop to scroll down the webpage
for i in tqdm(range(NO_OF_SCROLLS)):
    # Scroll down to the bottom of the page
    driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")

    # Wait for a short time to allow the page to load more content
    time.sleep(SCROLL_PAUSE_TIME)

    # Calculate the new scroll height after scrolling
    new_height = driver.execute_script("return document.body.scrollHeight")

    # Check if the new scroll height is the same as the last scroll height
    # If it is the same, it means that the page has reached the end, so break the loop
    if new_height == last_height:
        break

    # Update the last scroll height to the new scroll height for the next iteration
    last_height = new_height
```

### Step 4: HTML Parsing, Link Extraction, and Property Specification Extraction
Once the scrolling process has been completed, the script will get all of the HTML content from the webpage with the Selenium WebDriver and then parse it using the superb Beautiful Soup library. The script will then take the relevant property listing element and extract all the property links from within it. It then commences a series of iterations by accessing every extracted property link. At each link, it extracts from the corresponding HTML content of that property the whole array of all specifications about the property, including property ID, title, description, features, station distances, and any other relevant information.

```python
# Get the HTML content of the page using Selenium WebDriver
html_content = driver.page_source

# Parse the HTML content using Beautiful Soup
soup = BeautifulSoup(html_content, 'html.parser')

# Find the element containing property listings
property_listings = soup.find(id='property-data')

# Extract all property links from the property listings
all_links = [propert['href'] for propert in property_listings.find_all('a', class_='pli clearfix')]

all_links = [link for link in all_links if len(link) == 8]

# Initialize an empty list to store all property specifications
all_specs = []

# Iterate through all property links
for link in tqdm(all_links):
    # Get the HTML content of the property page
    page = requests.get(f'https://www.openrent.co.uk{link}').text.replace('\r', '').replace("\n", '').replace('\xa0', ' ')

    # Parse the HTML content using Beautiful Soup
    soup = BeautifulSoup(page, 'html.parser')

    # Initialize a dictionary to store specifications of the current property
    spec_dict = {}

    # Extract property ID from the link
    spec_dict['id'] = int(link[1:])

    # Extract property title and description
    spec_dict['title'] = soup.find('h1').text
    spec_dict['description'] = soup.find(class_='description').text

    # Extract property features and their values
    for stat in soup.find_all('table')[0].find_all("td"):
        feature = stat.find("span").text.strip()[:-1]
        value = stat.find('strong').text.strip()
        spec_dict[feature] = value

    # Extract additional features and their values
    for table in soup.find(id='FeaturesTab').find_all('table'):
        for row in table.find_all('tr'):
            feature = row.find_all('td')[0].text.strip()
            if len(row.find_all('td')[1].find_all('i')) > 0:
                if row.find_all('td')[1].find_all('i')[0]['class'][1] == 'fa-check':
                    value = 'Yes'
                else:
                    value = 'No'
            else:
                value = row.find_all('td')[1].text.strip()
            spec_dict[feature] = value

    # Extract station distances
    station_distances = []
    try:
      for row in soup.find(id='LocalTransport').find_all('tr')[1:]:
          station = row.find_all('td')[1].text.strip()
          distance = row.find_all('td')[2].text.strip()
          station_distances.append([station, distance])
      spec_dict['station_distances'] = station_distances
    except AttributeError:
      pass
    # Append the property specifications to the list
    all_specs.append(spec_dict)
```

### Step 5: Data Storage and Google Drive Integration

When iterating over the property links, the script accumulates the specified details in a list of dictionaries, and eventually converts it to a Pandas DataFrame. It then associates with Google Drive so that it can instantly save the resulting DataFrame as a CSV file on the user's Drive. The CSV file is named by the user-selected city for ease of identification and data sorting during the next step of processing. This comprehensive web scraping script serves as a powerful tool for data collection and preparation, laying the foundation for training a robust house prediction model. The scraped data can be further pre-processed, cleaned, and augmented with additional relevant features to train machine learning and deep learning models that accurately predict house prices or other pertinent target variables. Ensemble techniques mentioned include the combination of multiple models, where each model harnesses the strengths of each approach, in turn augmenting prediction accuracy and robustness. But before that, we scrape the house images as well.

```python
# Create a DataFrame from the list of property specifications
df = pd.DataFrame.from_dict(all_specs, orient='columns')

# Display the DataFrame
df
```
The final scraped data looks like this
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>description</th>
      <th>Bedrooms</th>
      <th>Bathrooms</th>
      <th>Max Tenants</th>
      <th>Location</th>
      <th>Deposit</th>
      <th>Rent PCM</th>
      <th>Bills Included</th>
      <th>Broadband</th>
      <th>Student Friendly</th>
      <th>Families Allowed</th>
      <th>Pets Allowed</th>
      <th>Smokers Allowed</th>
      <th>DSS/LHA Covers Rent</th>
      <th>Available From</th>
      <th>Online Viewings</th>
      <th>Garden</th>
      <th>Parking</th>
      <th>Fireplace</th>
      <th>Furnishing</th>
      <th>EPC Rating</th>
      <th>DSS Income Accepted</th>
      <th>EPC Not Required</th>
      <th>Students Only</th>
      <th>Maximum Tenancy</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1957557</td>
      <td>2 Bed Flat, Albion Gate, G1</td>
      <td>Smart 2x doubl...</td>
      <td>2</td>
      <td>1</td>
      <td>4</td>
      <td>Glasgow</td>
      <td>£1,600.00</td>
      <td>£1,400.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Today</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>Furnished</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1980803</td>
      <td>2 Bed Flat, Blackfriars Road, G1</td>
      <td>Two bedroom fl...</td>
      <td>2</td>
      <td>1</td>
      <td>2</td>
      <td>Glasgow</td>
      <td>£1,400.00</td>
      <td>£1,400.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>25 March, 2024</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>B</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011373</td>
      <td>Room in a Shared Flat, Bothwell Street, G2</td>
      <td>AVAILABLE 14th...</td>
      <td>2</td>
      <td>2</td>
      <td>1</td>
      <td>Glasgow</td>
      <td>£900.00</td>
      <td>£600.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>Today</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>NaN</td>
      <td>Yes</td>
      <td>Shared Accommodation</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1955586</td>
      <td>2 Bed Flat, Argyle Street, G2</td>
      <td>This apartment...</td>
      <td>2</td>
      <td>2</td>
      <td>2</td>
      <td>Glasgow</td>
      <td>£1,370.00</td>
      <td>£1,370.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Today</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

## House Image Scraping

The script begins by importing the libraries required for its functioning. These include time for introduction of strategic delays, PIL for image processing operations, tqdm for the creation of visually pleasing progress bars, requests for the handling of HTTP requests, Pandas for working with tabular data structures, Beautiful Soup for parsing HTML content, and os for interacting with the operating system's file system. The script then sets display.max_columns in Pandas such that all columns are displayed in the resulting DataFrame.

It also imports the drive module from the google.colab library, which is used to mount the user's Google Drive within the Google Colab environment.

### Step 1: Data Loading and Preparation

It then mounts the user's Google Drive so that files are accessible from the Drive. The CSV file "Openrent Liverpool.csv" is then read from the user's Google Drive at the designated location. The data is loaded into a Pandas Data Frame named, and the unique property IDs are extracted from the DataFrame's 'id' column and saved in a different list named id_list.
```python
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

df = pd.read_csv('/content/drive/MyDrive/Project/House_Prices_Scraped_Data/Openrent Liverpool.csv')
df
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>id</th>
      <th>title</th>
      <th>description</th>
      <th>Bedrooms</th>
      <th>Bathrooms</th>
      <th>Max Tenants</th>
      <th>Location</th>
      <th>Deposit</th>
      <th>Rent PCM</th>
      <th>Bills Included</th>
      <th>Broadband</th>
      <th>Student Friendly</th>
      <th>Families Allowed</th>
      <th>Pets Allowed</th>
      <th>Smokers Allowed</th>
      <th>DSS/LHA Covers Rent</th>
      <th>Available From</th>
      <th>Minimum Tenancy</th>
      <th>Online Viewings</th>
      <th>Garden</th>
      <th>Parking</th>
      <th>Fireplace</th>
      <th>Furnishing</th>
      <th>EPC Rating</th>
      <th>DSS Income Accepted</th>
      <th>Maximum Tenancy</th>
      <th>EPC Not Required</th>
      <th>Students Only</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2011776</td>
      <td>Studio Flat, Vista Residence, L2</td>
      <td>We are proud t...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Liverpool</td>
      <td>£200.00</td>
      <td>£815.00</td>
      <td>Yes</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Today</td>
      <td>6 Months</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1967713</td>
      <td>1 Bed Flat, Cumberland Street, L1</td>
      <td>We are proud t...</td>
      <td>1</td>
      <td>1</td>
      <td>1</td>
      <td>Liverpool</td>
      <td>£951.92</td>
      <td>£825.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Today</td>
      <td>6 Months</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>B</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2011825</td>
      <td>2 Bed Flat, Cheapside, L2</td>
      <td>APARTMENT IS A...</td>
      <td>2</td>
      <td>2</td>
      <td>3</td>
      <td>Liverpool</td>
      <td>£1,269.23</td>
      <td>£1,100.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>11 March, 2024</td>
      <td>6 Months</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>D</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1897891</td>
      <td>1 Bed Flat, Mathew Street, L2</td>
      <td>Self Service A...</td>
      <td>1</td>
      <td>1</td>
      <td>4</td>
      <td>Liverpool</td>
      <td>£1,000.00</td>
      <td>£2,300.00</td>
      <td>Yes</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>NaN</td>
      <td>Today</td>
      <td>1 Months</td>
      <td>NaN</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>Furnished</td>
      <td>D</td>
      <td>Yes</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>4</th>
      <td>1987164</td>
      <td>1 Bed Flat, Cumberland Street, L1</td>
      <td>A well present...</td>
      <td>1</td>
      <td>1</td>
      <td>2</td>
      <td>Liverpool</td>
      <td>£925.00</td>
      <td>£825.00</td>
      <td>No</td>
      <td>View Offers</td>
      <td>Yes</td>
      <td>Yes</td>
      <td>No</td>
      <td>No</td>
      <td>No</td>
      <td>21 March, 2024</td>
      <td>6 Months</td>
      <td>Yes</td>
      <td>No</td>
      <td>Yes</td>
      <td>No</td>
      <td>Furnished</td>
      <td>C</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>

### 4.2.2 Image Downloading Process

It then enters a loop that iterates through each property ID in the id_list. For each ID, it constructs the corresponding URL for the property page on the OpenRent website and retrieves the HTML content of that page using the requests library.

After retrieving the HTML content, the model uses Beautiful Soup to parse the HTML and extract the URLs of all thumbnail images associated with the property listing. These image URLs are stored in a list called image_links followed by creating a new folder within the user's Google Drive using the property ID as the folder name. This folder will be used to store the downloaded images for the corresponding property.

Within the newly created folder, the script iterates through each image URL in the image_links list. For each image URL, it constructs the complete URL by prepending "https:" to the relative URL. It then sends a GET request to the constructed image URL using the requests library and checks if the request was successful (status code 200). If the request was successful, it opens a new file in binary write mode within the property folder, with a filename derived from the image's index in the image_links list (e.g., "0.jpg", "1.jpg", etc.). The script then writes the content of the response (the image data) to the file, effectively saving the image.

If the request fails, the script prints a message indicating that the image download failed for the particular property ID. It continues this process for all image URLs associated with the property, ensuring that all available images are downloaded and saved within the corresponding property folder.


```python
id_list = df['id'].values

import requests

for id in tqdm(id_list):

  page = requests.get(f'https://www.openrent.co.uk/{id}').text.replace('\r', '').replace("\n", '').replace('\xa0', ' ')

  # Parse the HTML content using Beautiful Soup
  soup = BeautifulSoup(page, 'html.parser')

  image_links = [k['href'] for k in soup.find_all(class_ = "photos thumbnail mfp-image")]

  work_folder = f"/content/drive/MyDrive/Project/" + str(id)
  os.mkdir(work_folder)

  for num,image_link in enumerate(image_links):

    url = f"https:{image_link:3}"

    # Preferred name and location to save the image
    file_name = f"{num}.jpg"  # Change the preferred name as desired
    save_path = f"{work_folder}/" + file_name  # Change the save location as desired

    # Send a GET request to the URL
    response = requests.get(url)

    # Check if the request was successful (status code 200)
    if response.status_code == 200:
        # Open the file in binary write mode and write the content of the response
        with open(save_path, 'wb') as f:
            f.write(response.content)
    else:
        print("Failed to download the image: ",id)
```


## Image Processing

Next, in another notebook, we develop a script that processes images associated with house listings, aiming to create a dataset suitable for training a machine learning model.

First, we import the necessary libraries:
```python
import pandas as pd  # Importing pandas library and aliasing it as pd
import numpy as np  # Importing numpy library and aliasing it as np
from matplotlib import pyplot as plt  # Importing pyplot module from matplotlib library and aliasing it as plt
import os  # Importing os module for operating system dependent functionality
import seaborn as sns  # Importing seaborn library and aliasing it as sns
import warnings
import cv2
from tqdm import tqdm

warnings.filterwarnings('ignore')
pd.set_option('display.max_columns', None)   # Setting pandas option to display all columns in DataFrame
plt.style.use('ggplot')  # Setting plot style to 'ggplot' from matplotlib
```
### Loading and Cleaning Data
```python
df = pd.read_csv('/content/drive/MyDrive/House_Prices_Scraped_Data/Images Annotation - Final Doc.csv')
df
```
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>House ID</th>
      <th>Bedroom</th>
      <th>Bathroom</th>
      <th>Kitchen</th>
      <th>Sitting Room</th>
      <th>Frontage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1125181</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1225446</td>
      <td>14</td>
      <td>18</td>
      <td>3</td>
      <td>10</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1956767</td>
      <td>14</td>
      <td>15</td>
      <td>9</td>
      <td>10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1948597</td>
      <td>23</td>
      <td>27</td>
      <td>14</td>
      <td>4</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003468</td>
      <td>28</td>
      <td>20</td>
      <td>2</td>
      <td>12</td>
      <td>48</td>
    </tr>
  </tbody>
</table>

### Step 2: Cleaning the data

```python
# Removing rows where 'House ID' is null
df = df[~df['House ID'].isnull()]

# Dropping duplicate rows based on 'House ID'
df.drop_duplicates(['House ID'], inplace=True)

# Replacing 'Null' and 'N' values with NaN
df.replace(['Null', 'N'], np.NaN, inplace=True)
```

### Step 3: Summarizing missing values
```python
# Counting missing values in each column
non_missing_values_count = df.shape[0] - df.isnull().sum()

# Calculating the proportion of missing values for each column
non_missing_values_proportion= df.isnull().sum() / len(df)

# Combining count and proportion into one DataFrame for a clean summary
non_missing_values_summary = pd.DataFrame({
    'NOn Null Values': non_missing_values_count,
    'Proportion': non_missing_values_proportion
})
# Displaying the summary table
print(non_missing_values_summary)
```

| Columns | Non Null Values | Proportion |
| -------- |  --------  |  -------  |
|House ID|  3555 | 0.000000 |
| Bedroom | 3107 | 0.126020 |
| Bathroom | 3003 | 0.155274 |
| Kitchen | 2923 | 0.177778 |
| Sitting Room | 2322 | 0.346835 |
| Frontage | 1536 | 0.567932 |

This shows that the region with the least images is the frontage.

### Step 4: Data filtering

For the model to work optimally, it needs features with complete information. Therefore, this notebook defines a list of features and then filters the data to include only entries with complete information for these features. This ensures a dataset where each house has a corresponding image for each selected feature.

```python
## Selecting Rows with Complete Information for Selected Columns

# List of columns to consider
interested_cols = ['Bedroom', 'Bathroom', 'Kitchen', 'Sitting Room']

# Filtering DataFrame to exclude rows with any null values in interested columns
main_df = df[~df[interested_cols].isnull().any(axis=1)]

# Resetting index to ensure consecutive integer index after filtering
main_df.reset_index(drop=True, inplace=True)

# Displaying the filtered DataFrame
main_df
```

<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>House ID</th>
      <th>Bedroom</th>
      <th>Bathroom</th>
      <th>Kitchen</th>
      <th>Sitting Room</th>
      <th>Frontage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>1125181</td>
      <td>8</td>
      <td>3</td>
      <td>4</td>
      <td>0</td>
      <td>1</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1225446</td>
      <td>14</td>
      <td>18</td>
      <td>3</td>
      <td>10</td>
      <td>24</td>
    </tr>
    <tr>
      <th>2</th>
      <td>1956767</td>
      <td>14</td>
      <td>15</td>
      <td>9</td>
      <td>10</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>3</th>
      <td>1948597</td>
      <td>23</td>
      <td>27</td>
      <td>14</td>
      <td>4</td>
      <td>29</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2003468</td>
      <td>28</td>
      <td>20</td>
      <td>2</td>
      <td>12</td>
      <td>48</td>
    </tr>
  </tbody>
</table>

### Image Merging Functions
**Adding Borders:** The script defines a function `add_border` that takes an image, border color, and border size as input. It adds a border around the image using the Pillow (PIL) library's `ImageOps` module.

**Merging Images:** Another function,  `merge_images`, takes several arguments:

-   `base_folder`: The location where the house images are stored.
-   `image_paths`: A list containing filenames of the images to be merged
-   `output_name`: The filename for the resulting merged image.
-   `target_size`: The desired size for each image after resizing.
-   `border_color`: The color for the border added around each image.
-   `border_size`: The width of the border. The function resizes and adds borders to the specified images before merging them into a single image with a predefined layout. This creates a combined image representation for each house, potentially containing all available images of its features.

```python
# Importing necessary module
from PIL import Image, ImageOps

# Function to add a border around the given image
def add_border(image, border_color, border_size):
    """
    Add a border around the given image.
    """
    return ImageOps.expand(image, border=border_size, fill=border_color)

# Function to merge images and save the resulting image
def merge_images(base_folder, image_paths, output_name, target_size=(200, 200), border_color="white", border_size=5):
    # List to store resized and bordered images
    images = []

    # Load, resize, and add border to images
    for path in image_paths:
        img = Image.open(base_folder + '/' + path + '.jpg')
        img = img.resize(target_size)
        img = add_border(img, border_color, border_size)
        images.append(img)

    # Calculate the size of the resulting image
    result_width = target_size[0] * 2 + border_size * 2
    result_height = target_size[1] * 2 + border_size * 2

    # Create a blank image with a size sufficient to contain all the resized images
    result = Image.new("RGB", (result_width, result_height), border_color)

    # Paste resized images into the blank image
    for i in range(len(images)):
        x = i % 2
        y = i // 2
        result.paste(images[i], (x * target_size[0] + border_size, y * target_size[1] + border_size))

    # Save the resulting image
    result.save(output_name)
```
### Combining Images and Handling Errors
The script iterates through each row (house) in the filtered DataFrame. For each house, it:

1.  Extracts the house ID.
2.  Defines the base folder where the house's images are stored.
3.  Extracts a list of image paths based on the interested feature columns.
4.  Defines the output filename for the merged image.
5.  Attempts to merge the images using the `merge_images` function.
    -   If successful, a combined image is created for the house.
    -   If a `FileNotFoundError` occurs, it indicates missing images for that house. The script logs the house ID with the error and continues processing other houses.

```python
# List to store house IDs with errors
error_ids = []

# Iterating through each row in the DataFrame
for num in tqdm(range(main_df.shape[0])):
    # Extracting house ID from the DataFrame
    house_id = main_df.loc[num, 'House ID']

    # Defining the base folder where images are stored
    base_folder = f'/content/drive/MyDrive/House_Prices_Scraped_Data/Openrent Images/{house_id}'

    # Extracting image paths for interested columns
    image_paths = main_df.loc[num, interested_cols].values.tolist()

    # Defining the output name for the combined image
    output_name = f'/content/drive/MyDrive/House_Prices_Scraped_Data/Combined Images/{house_id}.jpg'

    try:
        # Merging images and saving the combined image
        merge_images(base_folder, image_paths, output_name, target_size=(480, 480), border_color="white", border_size=10)
    except FileNotFoundError:
        # Handling file not found error
        error_ids.append(house_id)
        print(f"File Found error: House ID {house_id} ")

```

The script encounters errors during the initial attempt to merge images for some houses. These errors likely results from missing image files. To address this, the script creates a list containing house IDs with these errors. It then isolates these problematic houses in a separate DataFrame. The script iterates through each house on this "error list" and tries to merge their images again. If successful, a combined image is created. However, if the `FileNotFoundError` persists, it indicates missing image files and the house ID is added back to the error list. This process essentially refines the error handling by focusing on houses with missing images and attempting to create merged images for them again.

```python
error_ids = ['2003463',
 '2008202',
 '2011088',
 '2013130',
 '1993243',
 '2011782',
 '1980041',
 '1964312',
 '2008197']

error_df = main_df[main_df['House ID'].isin(error_ids)].reset_index(drop = True)


for num in tqdm(range(error_df.shape[0])):
  house_id = error_df.loc[num,'House ID']
  base_folder = f'/content/drive/MyDrive/House_Prices_Scraped_Data/Openrent Images/{house_id}'

  image_paths = error_df.loc[num, interested_cols].values.tolist()

  output_name = f'/content/drive/MyDrive/House_Prices_Scraped_Data/Combined Images/{house_id}.jpg'
  try :
    merge_images(base_folder, image_paths, output_name, target_size=(480, 480), border_color="white", border_size=10)
  except FileNotFoundError:
    error_ids.append(house_id)
    print(f"File Found error: House ID {house_id} ")
```

### Generating List of Processed Houses

After processing all houses (including re-attempts for errors), the script creates a final list containing only house IDs with successfully processed images. 

```python
# Filtering out rows with error house IDs
non_error_df = main_df[~main_df['House ID'].isin(error_ids)].reset_index(drop=True)

# Saving the list of processed house IDs to a CSV file
non_error_df[['House ID']].to_csv('Processed Images Houses List.csv', index=False)

# Displaying the DataFrame without error entries
non_error_df
```

In the next part, we would explore the Machine Learning implementation.