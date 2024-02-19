# Coding Samurai Projects

## Task 1: Airbnb Exploratory Data Analysis (EDA)

### Overview

This project involves an in-depth Exploratory Data Analysis (EDA) of Airbnb datasets to uncover patterns, trends, and insights that can be valuable for hosts and guests alike. The analysis covers various aspects such as pricing distribution, property and room type preferences, geographical distribution, and sentiment analysis of guest reviews.

### Data Source

- [Dataset](https://www.kaggle.com/datasets/airbnb/seattle/data): Contains information about calendar, listings, and reviews.

### Key Findings

1. **Price Distribution Analysis:**
   - Visualized the distribution of listing prices, identifying common pricing trends.

2. **Property and Room Type Preferences:**
   - Explored the popularity of different property and room types among hosts and guests.

### Visualizations

#### 1. Price Distribution
![Price Distribution](https://drive.google.com/file/d/1KWUeJBgZR_vcTeB4tKx4o1LTUA71hsn3/view?usp=sharing)

#### 2. Property Type Preferences
![Property Type Preferences](https://drive.google.com/file/d/1p2w98r9LotxLjO3qZMYhQyjq6LI4Gg6K/view?usp=sharing)

## Project 2: Spam Email Detection

This project implements a Support Vector Machine (SVM) model for spam email detection using the Spambase dataset. The model achieved a notable accuracy of 73.72% on the test dataset. The implementation includes the training of the SVM model, feature engineering using TF-IDF vectorization, and the deployment of a Command Line Interface (CLI) for practical use.

### Files and Directory Structure

- **Dataset:**
  - `spambase.DOCUMENTATION`: Documentation for the Spambase dataset.
  - `spambase.data`: The Spambase dataset.
  - `spambase.names`: Information about the dataset.

### How to Use

1. Open `model_creation.ipynb` to view the model training and evaluation process.
2. Execute `spam_email.py` to use the SVM model through the CLI for predicting spam or ham emails.

### Acknowledgments

- The project uses the Spambase dataset. Refer to `spambase.names` for dataset details.
- The SVM model is serialized using joblib for ease of use and deployment.

Feel free to explore the code, contribute, and use this project as a foundation for Airbnb analysis and spam email detection applications.
