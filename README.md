# Crop Residue Management Optimizer

## Overview
This project helps predict crop residue types using soil and environmental data. The goal is to support farmers in making smart, data-driven decisions for managing crop residues, which can improve soil health and crop yield.

## What I Learned
- How soil and environmental factors affect crop residues.  
- How to use machine learning to make predictions.  
- How to preprocess data and train a model effectively.  
- How to create a self-contained notebook that runs without extra files.

## Tools & Technologies
- Python  
- Jupyter Notebook  
- Pandas & NumPy (for handling data)  
- Scikit-learn (Random Forest, train-test split, encoding)  
- Matplotlib & Seaborn (visualizations)

## Problem
Managing crop residues properly is a challenge for farmers. Poor management can affect soil health and reduce yield. We needed a solution to **predict the type of crop residue** based on soil and environmental conditions.

## Solution
- Collected a dataset with features like `Soil_pH`, `Moisture`, and `Temperature`.  
- Preprocessed the data and split it into training and testing sets.  
- Trained a Random Forest model to predict crop residue type.  
- Evaluated the model with accuracy and classification reports.  
- Embedded the dataset inside the notebook so it runs on its own without external files.

## Results
- The model gives **~90% accuracy** on the test set.  
- Key features affecting predictions are `Soil_pH` and `Moisture`.  
