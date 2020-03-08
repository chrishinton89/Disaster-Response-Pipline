# Disaster-Response-Pipline

## Summary of the Project

This project is a web application which classifies messages during a disaster to support a disaster relief agency. The demonstrates data engineering skills alongside data preparation and machine learning modelling to built a reliable and 

## Files in the Repository

The files within the repository are:

### (Folder)Data
- DisasterResponse.db - The database for the messages. Here is where all the messages data will be prepared and stored into.
- disaster_categories.csv - Original datasource of the categories (Provided by UDACITY)
- disaster_messages.csv - Original datasource of the messages (Provided by UDACITY)
- process_data.py - Python script which prepares, cleans, tokenises the data and stores the results in the disaster response database.

### (Folder)Models
- classifier.pkl - Data model prepared from the train_classifier.py below.
- train_classifier.py - Python script to use the prepared data, split, fit, train, tune and evaluate results and store as a pickle classifier for the web application.

### (Folder)App
- run.py - Python script to start the web app and produce the visualsiation from the dataset using the classifier when called.
### (Folder)Templates
  - go.html - Extention of the master.html which classifies the category of the message.
  - master.html - HTML templates for the application.

## How to Run the Project

1) Firstly run the following script in the project data directory: `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
2) Secondly run the following script in the model directory: `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`
3) Initialise the web application by running the following in the app directory: `python run.py`
4) Open http://0.0.0.0:3001/

## Overview of Application

