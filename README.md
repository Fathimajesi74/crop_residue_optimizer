Crop Residue Management using Machine Learning
📌 Project Overview

This project applies Machine Learning to predict crop residue outcomes based on soil, crop, and environmental factors. The goal is to help farmers and researchers optimize residue management for better soil health and sustainable farming practices.

🎯 Learning Objectives

Understand how data preprocessing improves ML model performance.

Train and evaluate models for classification tasks.

Interpret feature importance to know which factors influence outcomes most.

Build a simple ML-powered Streamlit app for predictions.

🛠️ Tools & Technologies Used

Python (pandas, numpy, matplotlib, seaborn)

scikit-learn (Random Forest, Label Encoding, Train-Test Split)

Jupyter Notebook (for step-by-step implementation)

Streamlit (for interactive prediction app)

GitHub (for version control and submission)

🔬 Methodology

Data Collection & Preparation – Crop residue dataset with soil and environmental features.

Preprocessing – Encoded categorical values (e.g., soil types, crop IDs), handled scaling.

Model Training – Trained multiple models; finalized Random Forest Classifier.

Evaluation – Used accuracy score and classification report.

Visualization – Feature importance graph for interpretability.

Deployment (Optional) – Streamlit app for user-friendly predictions.

❓ Problem Statement

Crop residue mismanagement leads to soil degradation, pollution, and reduced yields. Farmers often lack insights into which factors affect residue outcomes.

✅ Solution

We built an ML model that analyzes soil, crop, and environmental data to predict crop residue levels and provide insights into key influencing factors. This helps in making informed, data-driven decisions.

📊 Results

Achieved model accuracy of ~85–90% after preprocessing and parameter tuning.

Feature importance shows Soil pH and Moisture strongly influence predictions.

Streamlit app enables quick, real-time predictions.

📌 Conclusion

Machine Learning can significantly improve crop residue management by providing actionable insights. This approach supports sustainable farming and better resource utilization.

🚀 How to Run

Clone this repo:

git clone <repo-link>
cd <repo-folder>


Run the Jupyter Notebook:

jupyter notebook CropResidue.ipynb


Run the Streamlit app:

streamlit run app.py
