# CS433 - Machine Learning Project 2

## Authors (team: Team_BAK)
- Elif Kurtay
- Ernesto Bocini
- Abdullah Aydemir

## Sleep Stage Classification Project

Sleep stage classificition with ETH Zurich Sensory-Motor Systems Lab for Machine Learning course on EPFL.

The traditional method of sleep stage classification is performed by sleep experts who assign a sleep stage based on the visual inspection of Polysomnography (PSG) data. However, this method is intrusive and disturbing for the patients, making it hard for them to obtain their regular sleep state. Our motivation is thus to combine data collected from a piezoelectric sensor and a radar sensor to provide a non-invasive monitoring system that has accuracy as close as possible to the PSG, the gold standard. 

## File structure
#### Data folder:
- processed: includes a single .csv file for each patient's radar (Somnofy), mat (EMFIT), and PSG imputed sleep stage values indexed to their local timestamp.
- processed-w-raw: includes a single .csv file for each patient's radar (Somnofy), mat (EMFIT), and PSG imputed sleep stage values with an addition of their distance_mean, movement_mean, respiration_rate_mean, and signal_quality_mean from Somnofy; heart rate (hr), respiratory rate (rr), and activity (act) from EMFIT indexed to their local timestamp.
- notice.md: a notice for developers

#### Documents folder:
- literature review: includes papers related to the topic.
- Demographycs.csv: includes demographic and diagnosis information of a various of patients. Not all patient information available is included in the data set.
- Info_file.rtf: information about raw data provided by all sensors.
- project.pptx: a presentation summarizing the topic and findings of the project.
- report.pdf: the pdf of the project report file including explanations of methods and our results.

#### Scripts folder:
- Preprocessing.ipynb
  - File containing functions to load, process, and show finding (through plots) about the data analysis.
- MLModels.ipynb
  - File containing implementation and results of all ML models used on the processed data set which as Grid Search, Gradient Descent, KNN, Naive Bayes, and Decision Tree.
- CNN_1D.ipynb
  - File where the training set is used to find the best CNN model. 
- Experimentation.ipynb
  - File containing implementation and results of Decision Tree with raw features included and a KMeans implementation to try unsupervised learning.
- loader.py
  - File where functions relating to loading the patient data to tranfer it to the format given in the "data" folder in the repository.
- constants.py
  - File where certain constants used throughout the repository is stored such as a colection of participant IDs and path calculation method for reading the data.
- helpers.py
  - File that contains various helper functions for the project generally including loss, gradient, and accuracy computations.
- run.py
  - Main script - loading the best trained ML model (Naive Bayes) and using the test set to make predictions

## How to reproduce our results
The initial raw data received from the ETH Zurich lab is not shared in this repository as it is confidential. However, the processed versions of the data is included which is enough to re-run all models.

### Create the environment
Make sure your environment satisfies the following requirements:
- Python 3.7+
- NumPy module 
- matplotlib
- Python's scikit-learn, tensorflow, dtaidistance

### Run the code
From the root folder of the project

```shell
python run.py
```
