
from tkinter import messagebox
from tkinter import *
from tkinter import simpledialog
import tkinter
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tkinter import simpledialog
from tkinter import filedialog
import os
from sklearn.feature_extraction.text import CountVectorizer
import re
from string import punctuation
from nltk.corpus import stopwords
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import VotingClassifier
import pickle
from sklearn.naive_bayes import BernoulliNB
stop_words = set(stopwords.words('english'))


main = tkinter.Tk()
main.title("SELF OPINION MINING FOR FEEDBACK MANAGEMENT SYSTEM") #designing main screen
main.geometry("1300x1200")

global filename
global X, Y
global X_train, X_test, Y_train, Y_test
global cv
global XX
rank = {}
global cls
global nb_acc,hybrid_acc


def upload():
    global filename
    global rank
    rank.clear()
    train = pd.read_csv('dataset/rank.csv')
    for i in range(len(train)):
        ids = train.get_value(i,0,takeable = True)
        rank_value = train.get_value(i,1,takeable = True)
        rank[ids] = rank_value 
    filename = filedialog.askopenfilename(initialdir="dataset")
    text.delete('1.0', END)
    text.insert(END,filename+" loaded\n");

def checkInput(inputdata):
    option = 0
    try:
        s = float(inputdata)
        option = 0
    except:
        option = 1
    return option

def clean_doc(doc):
    tokens = doc.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [word for word in tokens if len(word) > 1]
    tokens = ' '.join(tokens) #here upto for word based
    return tokens

def Preprocessing():
    global X
    global Y
    X = []
    Y = []
    text.delete('1.0', END)
    train = pd.read_csv(filename)
    train['reviews.rating'] = train['reviews.rating'].fillna(0)
    two = 0
    three = 0
    four = 0
    five = 0
    for i in range(len(train)):
        ids = train.get_value(i,"id")
        ratings = train.get_value(i,"reviews.rating")
        review = train.get_value(i,"reviews.text")
        temp = review
        rank_value = rank.get(ids) 
        check = checkInput(review)
        if check == 1:
            review = review.lower().strip()
            review = clean_doc(review)
            arr = review.split(" ")
            msg = ''
            for k in range(len(arr)):
                word = arr[k].strip()
                if len(word) > 2 and word not in stop_words:
                    msg+=word+" "
            textdata = msg.strip()  #+" "+icon
            if two <= 500 and rank_value == 2:
                X.append(textdata+" "+str(int(ratings)))
                Y.append(rank_value)
                print(str(i)+"=="+str(ids)+"==="+str(ratings)+" == "+str(temp)+" "+str(rank_value))
            if three <= 500 and rank_value == 3:
                X.append(textdata+" "+str(int(ratings)))
                Y.append(rank_value)
                print(str(i)+"=="+str(ids)+"==="+str(ratings)+" == "+str(temp)+" "+str(rank_value))
            if four <= 500 and rank_value == 4:
                X.append(textdata+" "+str(int(ratings)))
                Y.append(rank_value)
                print(str(i)+"=="+str(ids)+"==="+str(ratings)+" == "+str(temp)+" "+str(rank_value))
            if five <= 500 and rank_value == 5:
                X.append(textdata+" "+str(int(ratings)))
                Y.append(rank_value)
                print(str(i)+"=="+str(ids)+"==="+str(ratings)+" == "+str(temp)+" "+str(rank_value))
            if rank_value == 2:
                two = two + 1
            if rank_value == 3:
                three = three + 1
            if rank_value == 4:
                four = four + 1
            if rank_value == 5:
                five = five + 1
    X = np.asarray(X)
    Y = np.asarray(Y)
    print(Y)
    text.insert(END,'Total reviews found in dataset : '+str(len(X))+"\n")
    

def generateModel():
    global cls
    text.delete('1.0', END)
    global XX
    global cv
    global X_train, X_test, Y_train, Y_test
    cv = CountVectorizer(analyzer='word',stop_words = stop_words, lowercase = True)
    XX = cv.fit_transform(X).toarray()
    '''
    max_fatures = 541
    tokenizer = Tokenizer(num_words=max_fatures, split=' ')
    tokenizer.fit_on_texts(X)
    XX = tokenizer.texts_to_sequences(X)
    XX = pad_sequences(XX)
    '''
    X_train, X_test, Y_train, Y_test = train_test_split(XX,Y, test_size = 0.13, random_state = 0)
    
    print(X_train[0])
    text.insert(END,'Total features extracted from reviews are  : '+str(X_train.shape)+"\n")
    text.insert(END,'Total splitted records used for training : '+str(len(X_train))+"\n")
    text.insert(END,'Total splitted records used for testing : '+str(len(X_test))+"\n") 

def buildClassifier():
    global nb_acc,hybrid_acc
    global cls
    text.delete('1.0', END)
    naiveBayes = BernoulliNB(binarize=5.5)
    naiveBayes.fit(X_train, Y_train)
    y_pred = naiveBayes.predict(X_test) 
    nb_acc = accuracy_score(Y_test,y_pred)*100 
    text.insert(END,"Naive Bayes Accuracy : "+str(nb_acc)+"\n\n")

    knn = KNeighborsClassifier() 
    dt = DecisionTreeClassifier(max_depth=500, min_samples_split=10,random_state=0)
    cls = VotingClassifier(estimators = [('dt', dt), ('knn', knn)], voting = 'soft')
    cls.fit(X_train, Y_train)
    y_pred = cls.predict(X_test) 
    hybrid_acc = accuracy_score(Y_test,y_pred)*100 
    text.insert(END,"Hybrid KNN & Decision Tree Accuracy : "+str(hybrid_acc)+"\n\n")
    f = open('cls.pckl', 'wb')
    pickle.dump(cls, f)
    f.close()


def predictRanking():
    '''
    f = open('cls.pckl', 'rb')
    cls = pickle.load(f)
    f.close()
    '''
    text.delete('1.0', END)
    global cls
    review = tf1.get()
    rating = tf2.get()
    textdata = review.strip()
    textdata = textdata.lower()
    textdata = clean_doc(textdata)
    arr = textdata.split(" ")
    msg = ''
    for k in range(len(arr)):
        word = arr[k].strip()
        if len(word) > 2 and word not in stop_words:
            msg+=word+" "
    textdata = msg.strip()
    textdata = textdata.strip()+" "+rating.strip()
    print(textdata)
    cv1 = CountVectorizer(vocabulary=cv.get_feature_names(),stop_words = "english", lowercase = True)
    test1 = cv1.fit_transform([textdata])
    output = cls.predict(test1.toarray())[0]
    text.insert(END,'REVIEW: '+tf1.get()+"\n")
    text.insert(END,'RATING: '+tf2.get()+"\n\n")
    text.insert(END,'Predicted Ranking for above review & rating is : '+str(output))
    

def graph():
    height = [nb_acc,hybrid_acc]
    bars = ('Naive Bayes Accuracy','Hybrid Algorithm Accuracy')
    y_pos = np.arange(len(bars))
    plt.bar(y_pos, height)
    plt.xticks(y_pos, bars)
    plt.show()       

font = ('times', 16, 'bold')
title = Label(main, text='Characterizing Customer Early Reviews')
title.config(bg='LightGoldenrod1', fg='medium orchid')  
title.config(font=font)           
title.config(height=3, width=120)       
title.place(x=0,y=5)

font1 = ('times', 12, 'bold')
text=Text(main,height=20,width=100)
scroll=Scrollbar(text)
text.configure(yscrollcommand=scroll.set)
text.place(x=100,y=300)
text.config(font=font1)


font1 = ('times', 12, 'bold')
uploadButton = Button(main, text="Upload Amazon Reviews Dataset", command=upload)
uploadButton.place(x=50,y=100)
uploadButton.config(font=font1)  

preButton = Button(main, text="Tokenization & Preprocessing", command=Preprocessing)
preButton.place(x=320,y=100)
preButton.config(font=font1) 

modelButton = Button(main, text="Generate Train Test Model with FS", command=generateModel)
modelButton.place(x=570,y=100)
modelButton.config(font=font1) 

classifierButton = Button(main, text="Build Classifier", command=buildClassifier)
classifierButton.place(x=860,y=100)
classifierButton.config(font=font1) 

l1 = Label(main, text='Enter Reviews')
l1.config(font=font1)
l1.place(x=50,y=150)

tf1 = Entry(main,width=40)
tf1.config(font=font1)
tf1.place(x=200,y=150)

l2 = Label(main, text='Enter Rating')
l2.config(font=font1)
l2.place(x=50,y=200)

tf2 = Entry(main,width=40)
tf2.config(font=font1)
tf2.place(x=200,y=200)

runButton = Button(main, text="Run", command=predictRanking)
runButton.place(x=100,y=250)
runButton.config(font=font1)

graphButton = Button(main, text="Comparison Graph Naive Bayes & Hybrid Algorithm", command=graph)
graphButton.place(x=170,y=250 )
graphButton.config(font=font1)



main.config(bg='OliveDrab2')
main.mainloop()
