# application overview
This is a search,recommendation and classification web application built using python 3.7
  Search: 
    user can enter a string and click on search, app returns the top 10 most matching 
    documents in result.
  
  Recommendation: 
    user can select a document and then hit on the recommend button besides that 
  and application will return the top 10 recommended documents.
  
  Classification: 
    user can enter the details in the web form and get to see the prediction based on
   input provided. Model metrics can be seen by clicking on the metrics tab.

# Installation steps
1. Clone the folder from the git hub to your local system.
2. Open the cloned project in pycharm or any other IDE supporting python.
3. Make sure the python interpreter is set to  3.7 version.
4. Open python terminal and install nltk using pip install nltk.
5. After intallation type import nltk in python terminal and hit enter.
6. Type nltk.download() and hit enter , this will open a wizard showing all. 
   libraries in nltk package. Install stopwords, punkt and porter_test.
7. Locate the application.py file and execute it.

# Deployment steps
1. Install git on your local windows/MAC machine.
2. Create an account on heroku cloud.
3. Create a repository on heroku cloud.
4. Open command prompt and navigate to project root directory.
5. Enter below commands sequentially.
  a. heroku login.
  b. git init.
  c. git add .
  d. git commit -m "enter any commit message".
  e. git push heroku master.
  f. heroku open.
6. After the above steps the application is deployed and launched in browser.

# Libraries
1. re
2. random
3. time
4. math
5. csv
6. nltk.corpus import stopwords
7. textblob import TextBlob
8. nltk.stem import PorterStemmer
9. collections import defaultdict
10. flask import Flask , render_template , request
11. pickle

# references
1. https://www.analyticsvidhya.com/blog/2015/08/beginners-guide-learn-content-based-recommender-systems/ 
2. https://www.nltk.org/ 
3. https://janav.wordpress.com/2013/10/27/tf-idf-and-cosine-similarity/
4. https://pythonprogramming.net/lemmatizing-nltk-tutorial/ 
5. https://docs.python.org/2/library/pickle.html 
6. https://dzone.com/articles/naive-bayes-tutorial-naive-bayes-classifier-in-pyt 
7. https://en.wikipedia.org/wiki/Bayes%27_theorem 
8. https://machinelearningmastery.com/naive-bayes-for-machine-learning/ 

# Author
1. Abhay Singh



