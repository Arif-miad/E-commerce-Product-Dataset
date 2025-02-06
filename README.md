



# E-commerce Product Dataset project

## About the Dataset

This **synthetic dataset** represents e-commerce product data with 8000 unique entries. It simulates customer interactions on an e-commerce platform, capturing product details, customer demographics, purchase history, and review sentiment.

The dataset can be utilized for various machine learning tasks, such as **recommendation systems**, **customer segmentation**, **price prediction**, and **sentiment analysis**.

### Features:

- **Product_ID**: A unique identifier for each product.
- **Product_Name**: A name describing the product (e.g., "Wireless Mouse", "Smartphone"), generated according to its category.
- **Category**: The broad category the product belongs to (e.g., "Electronics", "Clothing", "Furniture").
- **Sub_Category**: A specific sub-category within the main category (e.g., "Mobile Phones" under "Electronics").
- **Price**: The price of the product, which varies based on its category.
- **Customer_Age**: The age of a customer who might purchase the product, ranging from 18 to 65 years.
- **Customer_Gender**: The gender of the customer (either "Male" or "Female").
- **Purchase_History**: A simulated count of the number of purchases made by the customer, influenced by their age and product category.
- **Review_Rating**: A rating given to the product, based on its price, ranging from 1 to 5 stars.
- **Review_Sentiment**: The sentiment of the review, which can be "Negative", "Neutral", "Positive", or "Very Positive", based on the price.

### Purpose

This dataset is designed to provide valuable insights for e-commerce businesses and can be used to explore a variety of machine learning applications, including:

- **Recommendation Systems**: Predicting products that a customer may be interested in based on their past purchases, age, gender, and review sentiment.
- **Customer Segmentation**: Grouping customers based on demographics, purchasing behavior, and product interactions.
- **Price Prediction**: Predicting product prices based on various features such as category, sub-category, and customer ratings.
- **Sentiment Analysis**: Analyzing customer sentiment from reviews based on the product price and other factors.

### Use Cases

1. **Recommendation Systems**:
    - Predict products a customer is likely to buy based on their demographics and previous interactions.
    - Example: If a customer has a history of purchasing electronic devices, recommend similar electronic products.

2. **Customer Segmentation**:
    - Group customers by age, gender, purchase history, and other attributes to understand behavior patterns.
    - Example: Segment customers into high-value or low-value groups to tailor marketing campaigns.

3. **Price Prediction**:
    - Build a model to predict the price of a product given its category, sub-category, customer age, and other features.
    - Example: Predicting the price of a product like a smartphone based on brand, customer demographics, and previous purchase patterns.

4. **Sentiment Analysis**:
    - Predict the sentiment of a customer review based on the price and rating of a product.
    - Example: Positive reviews might correlate with lower-priced products in popular categories.

### Data Format

The dataset is available in **CSV** format and includes 8000 records across the features listed above. The columns are:

- Product_ID
- Product_Name
- Category
- Sub_Category
- Price
- Customer_Age
- Customer_Gender
- Purchase_History
- Review_Rating
- Review_Sentiment

### Data Dictionary

| Column Name         | Description                                                                                                                                           | Data Type         |
|---------------------|-------------------------------------------------------------------------------------------------------------------------------------------------------|-------------------|
| Product_ID          | A unique identifier for each product.                                                                                                                  | Integer (ID)      |
| Product_Name        | Name of the product (e.g., "Wireless Mouse").                                                                                                          | String            |
| Category            | Broad category (e.g., "Electronics", "Clothing").                                                                                                      | String            |
| Sub_Category        | Specific sub-category (e.g., "Mobile Phones" under "Electronics").                                                                                     | String            |
| Price               | Price of the product (varies by category).                                                                                                            | Float             |
| Customer_Age        | Age of the customer.                                                                                                                                   | Integer           |
| Customer_Gender     | Gender of the customer ("Male" or "Female").                                                                                                           | String (Male/Female)|
| Purchase_History    | Number of purchases made by the customer.                                                                                                             | Integer           |
| Review_Rating       | Rating given to the product (1 to 5 stars).                                                                                                            | Integer (1-5)     |
| Review_Sentiment    | Sentiment of the review based on the price (Negative, Neutral, Positive, Very Positive).                                                              | String (Categorical)|

---

## Example Code Implementation

Here are a few examples of how you might work with this dataset in Python, using popular libraries such as `pandas`, `matplotlib`, `seaborn`, and `scikit-learn`.

### 1. Loading the Dataset

```python
import pandas as pd

# Load the dataset
df = pd.read_csv('ecommerce_dataset.csv')

# Show the first few rows
print(df.head())
```

### 2. Exploratory Data Analysis (EDA)

Performing some basic exploratory analysis, such as checking the distribution of prices and ratings.

```python
import matplotlib.pyplot as plt
import seaborn as sns

# Price distribution
plt.figure(figsize=(10, 6))
sns.histplot(df['Price'], kde=True, bins=50)
plt.title('Price Distribution')
plt.xlabel('Price')
plt.ylabel('Frequency')
plt.show()

# Rating distribution
plt.figure(figsize=(10, 6))
sns.countplot(x='Review_Rating', data=df)
plt.title('Review Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.show()
```

### 3. Sentiment Analysis of Reviews

You can use `scikit-learn` to perform a basic sentiment analysis based on the review sentiment.

```python
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# Preprocess data for sentiment analysis
df['Review_Sentiment'] = df['Review_Sentiment'].map({'Negative': 0, 'Neutral': 1, 'Positive': 2, 'Very Positive': 3})

X = df[['Price', 'Review_Rating']]  # Example features
y = df['Review_Sentiment']  # Target variable

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Random Forest model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy:.2f}')
```

### 4. Price Prediction Model

Here's a simple regression model to predict product prices.

```python
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error

# Select features and target
X = df[['Customer_Age', 'Purchase_History', 'Review_Rating']]  # Example features
y = df['Price']  # Target variable

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
print(f'Mean Absolute Error: {mae:.2f}')
```

---

## How to Use

1. **Clone this repository:**

```bash
git clone https://github.com/your-username/ecommerce-dataset.git
```

2. **Install required libraries:**

```bash
pip install pandas scikit-learn matplotlib seaborn
```

3. **Run the Python code for analysis** in your local environment.

---

## Links

- **Kaggle Notebook**: [Link to Kaggle Notebook](https://www.kaggle.com/code/arifmia/e-commerce-product-data-analysis-and-insights)
- **LinkedIn Profile**: [Link to LinkedIn Profile](www.linkedin.com/in/arif-miah-8751bb217)

---

## Contributing

Feel free to fork the repository, contribute improvements, or submit issues for any bugs or feature requests. Pull requests are welcome!

