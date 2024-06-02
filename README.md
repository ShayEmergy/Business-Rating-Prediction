# Restaurant Rating Predictor

Welcome to the Restaurant Rating Predictor repository! This project investigates the factors that most affect a restaurant’s rating using a linear regression model.

## Table of Contents

- [Introduction](#introduction)
- [Dataset](#dataset)
- [Model](#model)
- [Results](#results)

## Introduction

This project aims to identify and analyze the key factors influencing restaurant ratings. By utilizing a linear regression model, we can understand how different variables contribute to the overall rating a restaurant receives.

## Dataset

The dataset used in this project includes various features such as:
- Restaurant details (name, location, etc.)
- User reviews
- Check-in information
- Photos
- Tips
- User information

The data is sourced from the Yelp Open Dataset: https://www.yelp.com/dataset

## Model

The linear regression model is implemented using [scikit-learn](https://scikit-learn.org/). The main steps include:
- Data cleaning and preprocessing
- Feature selection based on correlation analysis
- Model training
- Model evaluation

The model script (`train_model.py`) details the implementation of these steps.

### Significant Features
Based on the correlation analysis, the following features were selected for the model:
- has_bike_parking
- is_open
- latitude
- longitude
- price_range
- average_review_age
- average_review_length
- average_review_sentiment
- average_tip_length
- average_review_count
- average_number_years_elite

## Results

The model's performance was evaluated using the R-squared (R^2) metric.

- **R^2 Score**: 0.6687545616171118

### Sample Predictions
Here are the predictions for a sample of test data compared to the actual values:

| Predicted Rating | Actual Rating |
|------------------|---------------|
| 4.68             | 5.0           |
| 3.27             | 3.0           |
| 3.71             | 3.5           |
| 2.54             | 2.0           |
| 4.22             | 4.0           |
