# Phase 2: AI/ML Internship Tasks

Three notebooks from Phase 2 of the AI/ML Engineering internship at DevelopersHub Corporation. Each one solves a different kind of problem: text classification, customer churn prediction, and house price prediction using both numbers and images.

## Task 1: News Topic Classifier

`phase2task1/`

This trains a BERT model to read a news headline and guess which category it belongs to: World, Sports, Business, or Science/Tech. It uses the AG News dataset, trains on a smaller chunk of it so things run faster, and checks how well it did using accuracy and F1 score. At the end there's a simple Gradio app where you can type in a headline and see the prediction live.

**Tools used:** transformers, datasets, torch, gradio

## Task 2: Customer Churn Prediction

`phase2tasks2/`

This predicts whether a telecom customer is likely to leave the company or not, using the IBM Telco Customer Churn dataset. The data gets cleaned up first (fixing some bad values, dropping the customer ID column), then two models are trained and compared: Logistic Regression and Random Forest. The Random Forest gets tuned to find its best settings, and a confusion matrix shows how accurate the final model is. The trained model is also saved so it can be reused later.

**Tools used:** scikit learn, pandas, joblib

## Task 3: House Price Prediction Using Both Data and Images

`phase2tasks3/`

This one predicts house prices by combining regular house details (size, bedrooms, bathrooms, age) with features pulled from house images using a pretrained ResNet18 model. Both sets of features are combined and fed into a Linear Regression model, then evaluated using MAE and RMSE.

Note: the data here is randomly generated for demonstration, not real house listings or real photos. The goal is to show how tabular data and image data can be combined in one model.

**Tools used:** torch, torchvision, scikit learn, pandas

## How to Run

Each notebook installs what it needs in the first cell, so no separate setup is required. Just open the folder and run the notebook:

```bash
cd phase2task1
jupyter notebook
```

Do the same for phase2tasks2 and phase2tasks3. Just run the cells in order from top to bottom and everything works on its own. Task 2 pulls its data straight from GitHub, Task 1 pulls its data from the datasets library and Task 3 creates its own sample data inside the notebook.

