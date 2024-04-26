## End to End Project

### Step 1: Create a new environment

```
conda create -p venv python==3.10.11

conda activate venv/
```
### Step 2: Create a .gitignore file

```
create the file by right click and include the venv in it
```

### Step 3: Create a requirements.txt file 
```
pip install -r requirements.txt
or
create the file by right clicking
```

### Step 4: Create a setup.py file 
```
This is to install the entire project as a package. Additionally, write a function to read the packages from requirements.txt
```

### Step5: Create a folder `src` 
```
Include exception, logger, and utils python files. Make this folder as a package by including __init__.py file. 
The src folder will include 2 folders with name components and pipelines, including  __init__.py also in those folders.
```
#### Step 5.1 Create a folder `components`

```
Include data_ingestion, data_transformation, model trainer, and __init_.py. These components are to be interconnected in future. 
```
#### Step 5.2 Create a folder called `pipeline`
```
Create two python files training_pipeline and prediction_pipeline with __init__.py folder
``` 

### Step 6: Create a folder called `notebooks` 
```
Create a folder called data and include the dataset. Additionally, create a EDA.ipynb file to do the EDA analysis. For training the model_training file is created.
```
### Step 6: Create a app.py 
```
This Flask app hosts a homepage and a prediction endpoint. The homepage ('/') displays an HTML template, while the prediction endpoint ('/predict') handles form submissions, predicts outcomes based on the data provided, and displays the results.
indexpage: http://127.0.0.1:5000
formpage: http://127.0.0.1:5000/predict

```
### Step 7: Create a folder called `templates` 
```
This folder contain the app's form.html, index.html,result.html
form.html consist the machine learning features to be taken into as input in order to predict the output
index.html consist the first page indicating the cloth website using for prediction
```