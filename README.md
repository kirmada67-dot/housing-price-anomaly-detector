# Housing Price Anomaly Detector
First ML project. Built on foundational knowledge of ML. An ML-based system that predicts house prices and classifies properties as fairly priced, underpriced, or overpriced. 
Based on a housing dataset sourced from Kaggle (https://www.kaggle.com/datasets/yasserh/housing-prices-dataset/data?select=Housing.csv)

## Problem Statement
It is designed for individuals or families who are looking to buy or rent housing properties and want a better understanding of pricing before making a decision.

## Working
The system takes inputs such as area, number of rooms, bathrooms, stories, and other property features, along with the actual listed price of the property.
Based on this, the model predicts an expected fair price using historical housing data, and then compares it with the actual listing price to classify the property as fairly priced, slightly/strongly overpriced, or slightly/strongly underpriced.
1. Takes property features (area, rooms, etc.) + listing price  
2. Predicts expected price using a trained ML model  
3. Compares predicted vs actual price  
4. Classifies property as:
   - fair
   - slightly/strongly underpriced
   - slightly/strongly overpriced

## Features
- Price prediction using Linear Regression (Baseline Model)  
- Cross-validation for model evaluation  
- RMSE-based threshold for decision making  
- Percentage-based intensity classification  
- Interpretable outputs (price gap and % difference)

## Tech Stack
- Python
- NumPy, Pandas
- scikit-learn

## Changelog

### v1.0 - Baseline Model (Current)
- Implemented Linear Regression for price prediction
- Applied cross-validation to evaluate model performance
- Built RMSE-based threshold for pricing decisions
- Classified properties as fair, underpriced, or overpriced
- Added percentage-based intensity (slightly / strongly)


