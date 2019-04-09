#
# Author: Abhay Singh
# DataMining : Project Phase 1
# UTA ID : 1********9
# Application written in python for performing text search based on tf-idf concept.
#


import re
import os
import time
import math
import csv
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from collections import defaultdict
from flask import Flask , render_template,request
import pickle


idfdictionary_global = {}
tfidfquerytermsdictionary_global = {}
documentindexedtfidfscoreofcorpus_global = {}
textual_reviews = {}
idfdictionary_global = {}
textual_reviews = {}
documentCount = 0
porter = PorterStemmer()


application = Flask(__name__)

'''
    Description : Function storingreviewsindictionary reads the csv file line by line and updates the dictionary
                  with line number as key and value as description of the reviews. This gives the application
                  flexibility to use the in memory object rather then referring to csv file again and again
                  Dictionary object is serialised and stored in textual_reviews.pickle. Later on subsequent
                  request this file will be referred for loading back dictionary object
    Input       : None
    Output      : textual_reviews. This is a dictionary data-structure and it stores document index as the key and
                  text reviews as the value. This dictionary is kept global so that it can be accessed throughout the
                  application.
'''
def storingreviewsindictionary():
    global textual_reviews
    count = 0
    with open('amazon.csv', encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            textual_reviews.update({count: row[1]})
            count = count + 1


    pickle_out = open("textual_reviews.pickle", "wb")
    pickle.dump(textual_reviews, pickle_out)
    pickle_out.close()

    return textual_reviews


'''
    Description : Function parsingDescriptionAndStoringInDictionary reads the global text_reviews dictionary and 
                  by iterating through the keys it performs cleaning of the values in stages as mentioned below.
                  
                  stage 1: Removes the unwanted characters from the review by using a custom defined regular
                           expression. Textual review gets filtered.
                  stage 2: Converting the description from stage 1 to lowercase and removing stop words from
                           reviews using stopwords from nltk package 
                  stage 3: Converting the words present in the reviews to its root word using stemming.
                  
                  Entire data is stored in dictionary data in form of document index and value as the 
                  filtered value of review obtained from stages 1,2 and 3
                  
    Input       : None
    Output      : dictionarydata


'''

def parsingDescriptionAndStoringInDictionary():
    global textual_reviews
    dictionaryData={}
    documentCount = 0

    for key in  textual_reviews:
        stringValue=textual_reviews[key]

        # using RE to remove special characters in the string
        regexFreeData = re.sub(r"[-()\"#/@;:<>{}`''+=~|.!?,''[]", "", stringValue)

        # Splitting the description column based on space
        valueArray = re.split('\s+', regexFreeData.lower())

        # removing the stopwords from the array
        stop_words = set(stopwords.words('english'))
        updatedValueArray = [w for w in valueArray if not w in stop_words]

        #Performing lematization
        dictionaryData[documentCount]=performLematization(updatedValueArray)

        documentCount=documentCount + 1

    return dictionaryData,documentCount


@application.route('/')
def index():
    return render_template('index.html')


'''
    Description : This is the main routing function invoked by the container when /searchreviews is invoked.All the 
                  related dictionary data is created by commenting line 138 -149 initially before web application is
                  hosted on cloud platform. After the serialised pickle files are created , application will refer 
                  them directly. After getting the first request application will no longer refer to pickle files,
                  it will directly refer to the below mentioned global variables.All the below mentioned global variables
                  are initialised when app receives http request for first time
                  global idfdictionary_global :  Stores the mapping of term and  (frequency of term in a document / total terms in document) in dictionary. 
                  global tfidfquerytermsdictionary_global : Stores the mapping of terms in query to their frequency.
                  global documentindexedtfidfscoreofcorpus_global : Stores the mapping of terms to number of documents in which it is occurring in dictionary.
                  global textual_reviews : Stores the mapping  of document index and text review data in dictionary.
                
    Input         : Search query received from the end user.
    Output        : Returns the list of top 10 most similar documents.
'''

@application.route('/searchreviews')
def searchreviews():
    # Get the user query from the request
    searchquery =request.args.get('searchquery')
    print(searchquery)
    global idfdictionary_global
    global documentindexedtfidfscoreofcorpus_global
    global idfdictionary_global
    global textual_reviews

    if len(documentindexedtfidfscoreofcorpus_global) == 0:
        pickle_in = open("documentindexedtfidfscoreofcorpus.pickle", "rb")
        documentindexedtfidfscoreofcorpus_global = pickle.load(pickle_in)

    if len(idfdictionary_global) == 0:
        pickle_in = open("idfdictionary.pickle", "rb")
        idfdictionary_global = pickle.load(pickle_in)

    if len(textual_reviews) == 0:
        pickle_in = open("textual_reviews.pickle", "rb")
        textual_reviews = pickle.load(pickle_in)

    if len(idfdictionary_global) > 0 and len(documentindexedtfidfscoreofcorpus_global) > 0:


        tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery, idfdictionary_global)
        cosineresult = calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary,
                                                                         documentindexedtfidfscoreofcorpus_global)
    else:

        storingreviewsindictionary()

        # Converting the fetched data into dictionary
        dictionaryData, documentCount = parsingDescriptionAndStoringInDictionary()

        # Converting
        validtermfrequency = creatingVectorRepresentationOfDocument(dictionaryData)
        idfdictionary = calculatedocumentfrequency(documentCount)

        idfdictionary_global = idfdictionary
        documentcorpusindexedtfidf = computeTfIDFOfTheCorpus(idfdictionary, validtermfrequency)
        documentindexedtfidfscoreofcorpus_global = documentcorpusindexedtfidf
        tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery, idfdictionary)
        tfidfquerytermsdictionary_global = tfidfquerytermsdictionary

        cosineresult = calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary,
                                                                         documentcorpusindexedtfidf)
    return render_template('tedresult.html',results=cosineresult)


def performLematization(wordtokens):
    wordinflectedlist=[]
    for word in wordtokens:
        wordinflectedlist.append(porter.stem(word))
    return wordinflectedlist



'''
    Description : creatingVectorRepresentationOfDocument creates a term frequency dictionary which is normalised.
    Input       : dictionaryData containing terms and its frequency in a document.
    Output      : documentTermFrequencyDictionary contains document index as key and normalised term frequency as value
                  for each document.
'''


def creatingVectorRepresentationOfDocument(dictionaryData):
    documentTermFrequencyDictionary ={}

    for i in dictionaryData:
        arrayData = dictionaryData[i]    # one row of data in array form
        dict1 = dict.fromkeys(list(arrayData), 0)

        for j in range(len(arrayData)):
            dict1[arrayData[j]] += 1
        normalisedTermFrequencyDictionary = {}
        for n in dict1:

            normalisedTermFrequencyDictionary.update({n:float(dict1[n] /len(arrayData))})

        documentTermFrequencyDictionary.update({i: normalisedTermFrequencyDictionary})

    return documentTermFrequencyDictionary

'''
    Description : calculatedocumentfrequency calculates the inverted document frequency for each term present in each document.
                  Each row read from the csv file is returned as a list of strings .TextBlob is a Python (2 and 3) library for 
                  processing textual data.It provides a simple API for diving into common natural language processing (NLP) tasks
                  such as part-of-speech tagging,noun phrase extraction, sentiment analysis, classification, translation, and more.
                  After applying the textblob to list of terms , it is checked for presence of stop words. Later the resultant list
                  of words goes through stemming (to convert it into its root form). After this series of operation is completed
                  i.e is foreach term of every document we compute the number of documents in which the terms occur. Then we calculate
                  the idf score for each term.
    Input         : Document count (100K in this case)
    Output        : idfDictionary , containing terms and its IDF score. 

'''

def calculatedocumentfrequency(documentCount):
    idfDictionary = {}
    ps = PorterStemmer()
    stop_word = set(stopwords.words('english'))
    word_dict = defaultdict(set)
    with open('./amazon.csv', encoding="utf8") as input:
        oops = csv.reader(input)
        next(oops)
        for count, i in enumerate(oops):
            for insert in set(TextBlob(i[1].lower()).words) - stop_word:
                word_dict[ps.stem(insert)].add(count)
    a = dict(word_dict.items())
    idfDictionary = {key: (1 + math.log(documentCount / len(a[key]))) for key in a.keys()}

    pickle_out = open("idfDictionary.pickle", "wb")
    pickle.dump(idfDictionary, pickle_out)
    pickle_out.close()

    return idfDictionary


'''
    Description : computeTfIDFOfTheCorpus caclualtes the TF * IDF score of each term present in the document.
                  This calculation repeats for entire set of documents.
    
    Input       : idfdictionary contains the inverted document frequency  for each term of a document.
                  documentTermFrequencyDictionary: Contains the term frequency for each document
            
    Output      : documentindexedtfidfscoreofcorpus contained document number as index and value as
'''

def computeTfIDFOfTheCorpus(idfdictionary , documentTermFrequencyDictionary):

    documentindexedtfidfscoreofcorpus = {}
    docindex=0


    for docid in documentTermFrequencyDictionary.keys():
        tfidfscoreofcorpus = {}
        documentDictionary = documentTermFrequencyDictionary[docid]
        for terms in documentDictionary.keys():
            if terms in idfdictionary:
                tfidfscoreofcorpus.update({terms : (documentDictionary[terms] * idfdictionary[terms])})

        documentindexedtfidfscoreofcorpus.update({docindex: tfidfscoreofcorpus})
        docindex = docindex + 1

    pickle_out = open("documentindexedtfidfscoreofcorpus.pickle", "wb")
    pickle.dump(documentindexedtfidfscoreofcorpus, pickle_out)
    pickle_out.close()

    return documentindexedtfidfscoreofcorpus


'''
    Description :convertQueryIntoDictionary manipulates the search query initiated by the end user. First a regular
                 expression is used to filter out unwanted characters. Then the resultant string is tokenised and s
                 stored in 'arrayQuery'. Then stop words are removed and lemmatisation is performed on the query tokens
                 Then there is calculation for term frequency of the tokens present in the query . Then there is
                 calculation for tf * idf score of the query terms
    
    Input       : idfdictionary ,Contains the inverted document frequency score for each term of a document.
                  query ,User search query 
    Output      : tfidfquerytermsdictionary, contains the TF * IDF score for each terms in the query
'''


def convertQueryIntoDictionary(query,idfdictionary):
    # Removing the special characters from the input user query
    regexFreeData = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", query)

    #print(regexFreeData.lower())
    arrayQuery = re.split("\s",regexFreeData.lower())

    stop_words = set(stopwords.words('english'))

    # updatedValueArray = removeStopWordsFromDescription(valueArray)
    updatedValueArray = [w for w in arrayQuery if not w in stop_words]

    filteredArrayQuery = performLematization(updatedValueArray)
    queryDictionary = dict.fromkeys(list(filteredArrayQuery), 0)

    for j in range(len(filteredArrayQuery)):
        queryDictionary[filteredArrayQuery[j]] += 1

    # queryDictionary contains terms and its frequency, now normalising the frequency.
    normalisedquerydictionary = {}

    for terms in queryDictionary.keys():
        freq = queryDictionary[terms] / len(queryDictionary)
        normalisedquerydictionary.update({terms:freq})

    #calculating the tf*idf of the query
    tfidfquerytermsdictionary={}

    for queryterms in normalisedquerydictionary.keys():
        if queryterms in idfdictionary.keys():
            tfidfquerytermsdictionary.update({queryterms:(normalisedquerydictionary[queryterms] * idfdictionary[queryterms])})

    return  tfidfquerytermsdictionary



'''

    Description : calculateCosineSimilarityBetweebQueryandDocuments caclulates the vector dot products between tfidfquerytermsdictionary 
                  and documentindexedtfidfscoreofcorpus. Top 10 documents are selected based on their cosine similarity value.
    Input       : tfidfquerytermsdictionary , Dictionary of query terms and their TF * IDF score
                  documentindexedtfidfscoreofcorpus , Dictionary having document index and value as idfdictionary for each document
    Output      : List of 10 documents having the most high cosine similarity value .  

'''

def calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary , documentindexedtfidfscoreofcorpus):

    cosineresultdictionary={}
    cosineList = []
    docindex = 0
    count = 0
    global textual_reviews

    for documentdictionary in range(len(documentindexedtfidfscoreofcorpus)):

        if docindex == len(documentindexedtfidfscoreofcorpus):
            break
        document = documentindexedtfidfscoreofcorpus[docindex]


        for queryterms in tfidfquerytermsdictionary.keys():

            if queryterms in document:
                cosineList.append([tfidfquerytermsdictionary[queryterms],document[queryterms]])


            else:
                cosineList.append([tfidfquerytermsdictionary[queryterms], 0])
                count = count +1

        # calculate the cosine similarity of the query and document at docindex

            num = 0.0
            totalmag = 0.0
            for product in range(len(cosineList)):
                num = num + cosineList[product][0] * cosineList[product][1]

            querydenominatormag = 0.0
            documentdenominatormag = 0.0

            if num != 0.0:
                for x in range(len(cosineList)):
                    querydenominatormag = querydenominatormag + cosineList[x][0] ** 2
                    documentdenominatormag = documentdenominatormag + cosineList[x][1] ** 2
                totalmag = math.sqrt(querydenominatormag * documentdenominatormag)

                cosineFactor = num / totalmag
                cosineresultdictionary.update({docindex: cosineFactor})


        docindex = docindex + 1
        cosineList.clear()
        count=0
    result = sorted(cosineresultdictionary.items(), key= lambda x: x[1] , reverse=True)

    list=[]
    returnresultcount = 0
    for k,v in result:

           r = textual_reviews[k]

           # read from textual_reviews

           list.append(r)
           returnresultcount = returnresultcount+1
           if returnresultcount == 10:
            break
    end = time.time()
    return list

if __name__ == '__main__':
   application.run()

