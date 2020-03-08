import sys
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn import multioutput
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import classification_report, precision_recall_fscore_support
import pickle

def load_data(database_filepath):
    '''
    Read database, define X&Y and category names.
    '''
    engine = create_engine('sqlite:///' + database_filepath)
    df = pd.read_sql_table('Messages', con=engine)
    X = df['message']
    y = df.iloc[:,4:]
    category_names = list(df.columns[4:])
    return X, y, category_names

def tokenize(text):
    ''' Function to replace characters
    '''
    # URL for list of all regular expressions
    url_regex = 'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+'
    # List for all URLS using regular expressions
    detected_urls = re.findall(url_regex, text)
    # For loop to replace each url with placeholder
    for url in detected_urls:
        text = text.replace(url, "urlplaceholder")
    # Tokenize
    tokens = word_tokenize(text)
    # Initiate Lemmatizer
    lemmatizer = WordNetLemmatizer()   
    # Iterate through each, lemmatize, normalize the case and remove spaces which trail or lead
    clean_tokens = []
    for tok in tokens:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    # Return cleaned data
    return clean_tokens

def build_model():
    '''
    Building model with tuned parameters.
    '''    
    pipeline = Pipeline([
        ('vect', CountVectorizer()),
        ('tfidf',TfidfTransformer()),
        ('clf', MultiOutputClassifier(RandomForestClassifier()))
        ])
    parameters = {'clf__estimator__min_samples_split': [2,10],
                  'clf__estimator__n_estimators':   [14],
                  'vect__max_df': (0.5, 0.75, 1.0)}
    model = GridSearchCV(pipeline, param_grid=parameters)

    return model

def evaluate_model(model, X_test, Y_test, category_names):
    '''
    Classification report to display results in terms of precision, recall and f1 score.
    '''
    y_pred = model.predict(X_test)
    print(classification_report(Y_test.iloc[:,1:].values, np.array([i[1:] 
        for i in y_pred])
            , target_names=category_names))


def save_model(model, model_filepath):
    '''
    Saving model as pickle file.
    '''
    path = model_filepath
    pickle.dump(model, open(path, 'wb'))
    pass

def main():
    '''
    Main function to initialise all other functions.
    '''
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()