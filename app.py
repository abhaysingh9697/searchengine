#
# Author: Abhay Singh
#
# This is a an application written in python for performing text search based on tf-idf concepts and cosine similarity.
#
#
#

import re
import os
import time
import math
import csv
import nltk
from nltk.corpus import stopwords
from textblob import TextBlob
from nltk.stem import PorterStemmer
from collections import defaultdict
from flask import Flask , render_template,request


idfdictionary_global = {}
tfidfquerytermsdictionary_global = {}
documentindexedtfidfscoreofcorpus_global = {}
idfdictionary_global = {}
textual_reviews = {}
documentCount = 0
porter = PorterStemmer()


app= Flask(__name__)

'''
    Description : Function storingreviewsindictionary reads the csv file line by line and updates the dictionary
                  with line number as key and value as description of the reviews. This gives the application
                  flexibility to use the in memory object rather then referring to csv file again and again
    Input       : None
    Output      : textual_reviews. This is a dictionary data-structure and it stores document index as the key and
                  text reviews as the value. This dictionary is kept global so that it can be accessed throughout the
                  application.
    
        
'''


def storingreviewsindictionary():
    global textual_reviews
    start = time.time()
    dictionary = {}
    count = 0
    with open('amazon.csv', encoding="utf8") as csvFile:
        reader = csv.reader(csvFile)
        next(reader)
        for row in reader:
            textual_reviews.update({count: row[1]})
            count = count + 1
    end = time.time()
    print("storingreviewsindictionary took {}".format(end - start))
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
    print("function : parsingDescriptionAndStoringInDictionary")
    start = time.time()
    global textual_reviews
    dictionaryData={}
    documentCount = 0

    for key in  textual_reviews:
        stringValue=textual_reviews[key]

        # using RE to remove special characters in the string
       # print(stringValue)
        regexFreeData = re.sub(r"[-()\"#/@;:<>{}`''+=~|.!?,''[]", "", stringValue)

        # Splitting the description column based on space
        valueArray = re.split('\s+', regexFreeData.lower())

        # removing the stopwords from the array
        stop_words = set(stopwords.words('english'))

        #updatedValueArray = removeStopWordsFromDescription(valueArray)
        updatedValueArray = [w for w in valueArray if not w in stop_words]

        #Performing lematization
        dictionaryData[documentCount]=performLematization(updatedValueArray)
        documentCount=documentCount + 1
        #print(documentCount,stringValue)


    end = time.time()
    print(end - start)
    return dictionaryData,documentCount


'''
    Description : index is the 
'''


@app.route('/')
def index():
    return render_template('index.html')


'''
    Description : This is the main routing function invoked by the container when /searchreviews

'''

@app.route('/searchreviews')
def searchtalks():
    # Get the user query from the request
    searchquery =request.args.get('searchquery')
    global idfdictionary_global
    global tfidfquerytermsdictionary_global
    global documentindexedtfidfscoreofcorpus_global
    global idfdictionary_global
    global textual_reviews

    #print( len(tfidfquerytermsdictionary_global) , len(documentindexedtfidfscoreofcorpus_global) )

    if len(tfidfquerytermsdictionary_global) > 0 and len(documentindexedtfidfscoreofcorpus_global) > 0:
        print("in the if block    ")

        tfidfquerytermsdictionary = convertQueryIntoDictionary(searchquery, idfdictionary_global)
        cosineresult = calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary,
                                                                         documentindexedtfidfscoreofcorpus_global)
    else:

        # Building the entire document indexing
        #dbquery="select REVIEWS from AMAZONPHONEREV"

        # Connect to IBM Cloud and fetch the Data
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

# def removeStopWordsFromDescription(arrayValues):
#     stopwordslist={'very','a','description','ourselves','hers','between','yourself','but','again','there','about','once','during','out','very','having','with','they','own','an','be','some','for','do','its','yours','such','into','of','most','itself','other','off','is','s','am','or','who','as','from','him','each','the','themselves','until','below','are','we','these','your','his','through','don','nor','me','were','her','more','himself','this','down','should','our','their','while','above','both','up','to','ours','had','she','all','no','when','at','any','before','them','same','and','been','have','in','will','on','does','yourselves','then','that','because','what','over','why','so','can','did','not','now','under','he','you','herself','has','just','where','too','only','myself','which','those','i','after','few','whom','t','being','if','theirs','my','against','by','doing','it','how','further','was','here','than'}
#   #  print(aamarrayValues)
#     for value in arrayValues:
#         if value in stopwordslist:
#             arrayValues.pop(arrayValues.index(value))
#             #print("removed stop word {}".format(value))
#
#     return arrayValues
#

def performLematization(wordtokens):
    wordinflectedlist=[]
    for word in wordtokens:
        wordinflectedlist.append(porter.stem(word))


    return wordinflectedlist


def creatingVectorRepresentationOfDocument(dictionaryData):
    #converting the query in the dictionary format
    print("function : creatingVectorRepresentationOfDocument")
    start = time.time()
    documentTermFrequencyDictionary ={}

    for i in dictionaryData:
        arrayData = dictionaryData[i]    # one row of data in array form
        dict1 = dict.fromkeys(list(arrayData), 0)

        for j in range(len(arrayData)):
            dict1[arrayData[j]] += 1
        # this ds has the document number as D2543 : Value as a dictionary which has the termfrequency count of the terms

        normalisedTermFrequencyDictionary = {}
        for n in dict1:

            normalisedTermFrequencyDictionary.update({n:float(dict1[n] /len(arrayData))})

        documentTermFrequencyDictionary.update({i: normalisedTermFrequencyDictionary})

        #Normalising the term count with respect to the number of valid terms in each document

   # print("documentTermFrequencyDictionary ----->{}".format(len(documentTermFrequencyDictionary)))
    end= time.time()
    print(end - start)
    return documentTermFrequencyDictionary




# Assembling an inverted index.It is a data structure that maps tokens to the documents they appear in.
def calculatedocumentfrequency(documentCount):
    idfDictionary = {}
    print("function : calculatedocumentfrequency")
    starttime = time.time()
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
    print(len(a.keys()))
    idfDictionary = {key: (1 + math.log(documentCount / len(a[key]))) for key in a.keys()}

    endtime = time.time()

    print("Time taken to index the corpus is {}".format(endtime - starttime))
    print("idfDictionary ----->{}".format(len(idfDictionary)))
    return idfDictionary


def computeTfIDFOfTheCorpus(idfdictionary , documentTermFrequencyDictionary):

    print("function : computeTfIDFOfTheCorpus")
    start = time.time()
    documentindexedtfidfscoreofcorpus = {}
    docindex=0
   # print(documentTermFrequencyDictionary)


    for docid in documentTermFrequencyDictionary.keys():
        tfidfscoreofcorpus = {}
       # print("iterating for document {}".format(docid))
        documentDictionary = documentTermFrequencyDictionary[docid]
       # print("document obtained is {}".format(documentDictionary))
        for terms in documentDictionary.keys():
            if terms in idfdictionary:
                tfidfscoreofcorpus.update({terms : (documentDictionary[terms] * idfdictionary[terms])})

       # print(docindex,tfidfscoreofcorpus)
        documentindexedtfidfscoreofcorpus.update({docindex: tfidfscoreofcorpus})
        docindex = docindex + 1

   # print(documentindexedtfidfscoreofcorpus)
    endtime=time.time()
    print(endtime - start)
    return documentindexedtfidfscoreofcorpus



def convertQueryIntoDictionary(query,idfdictionary):
    # Removing the special characters from the input user query

    print("function : convertQueryIntoDictionary")
    start = time.time()

    regexFreeData = re.sub(r"[-()\"#/@;:<>{}`+=~|.!?,]", "", query)
   # print(regexFreeData.lower())
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
   # print(tfidfquerytermsdictionary)

   # print(normalisedquerydictionary)
    end = time.time()
    print(end - start)
    return  tfidfquerytermsdictionary


def calculateCosineSimilarityBetweebQueryandDocuments(tfidfquerytermsdictionary , documentindexedtfidfscoreofcorpus):

    print("function : cosinesimilarity")
    start = time.time()
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

            if queryterms in documentindexedtfidfscoreofcorpus[docindex]:
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
           print("textual review {}".format(r))
           list.append(r)
           returnresultcount = returnresultcount+1
           if returnresultcount == 3:
            break
    end = time.time()
    print(end - start)
    print(list)
    return list


if __name__ == '__main__':
    app.run()