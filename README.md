# This is Capstone project for CPE312

Dataset used : [Early stage diabetes risk prediction dataset (Kaggle)](https://www.kaggle.com/datasets/yasserhessein/early-stage-diabetes-risk-prediction-dataset)

Before running ipynb file, ensure that all required Python packages are installed

or you can just run the following command :
```bash
!pip install -r requirements.txt
```

The following are the detail of each folders : 

- data : contain processed and raw folder
processed folder contain .csv file of cleaned data which is created after the process of wrangling.ipynb in src folder
raw folder contain .csv file of raw data

- models : contain .pkl files of trained Decision Tree, Logistic Regression, Random Forest models

- Results : contain figures and output folder
figures folder contain all visualization related to EDA process and model score used for comparison
output folder contain detailed chi-squared test result related to EDA process and some detailed model scores

- src : contain 2 main types of files which are .py and .ipynb
.ipynb files are related to all the process from wrangling > EDA > Training
.py files are used to test the classification of all trained model that are saved as .pkl files

To start running any process of .py files inside src folder make sure your current directory is at src
because the path used to reference .pkl files of model are relative paths

you can run .py file using the following command :
```bash
python filename.py
```
- LR_Predict.py for logreg_model.pkl which is Logistic Regression Model
- DT_Predict.py for dtree_model.pkl which is Decision Tree Model
- RF_Predict.py for randomforest_model.pkl which is Random Forest Model