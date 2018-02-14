# -*- coding: utf-8 -*-
"""
Created on Mon Feb 12 22:38:07 2018

@author: Vashi NSIT
"""

import pandas as pd
import numpy as np
import random as rnd

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC, LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.neural_network import MLPClassifier


from sklearn.cross_validation import train_test_split , StratifiedKFold
from sklearn.feature_selection import RFECV

import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.pylab as pylab

mpl.style.use('ggplot')
sns.set_style('white')
pylab.rcParams['figure.figsize']=8,6


def plot_distribution( df , var , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , hue=target , aspect=4 , row = row , col = col )
    facet.map( sns.kdeplot , var , shade= True )
    facet.set( xlim=( 0 , df[ var ].max() ) )
    facet.add_legend()

def plot_categories( df , cat , target , **kwargs ):
    row = kwargs.get( 'row' , None )
    col = kwargs.get( 'col' , None )
    facet = sns.FacetGrid( df , row = row , col = col )
    facet.map( sns.barplot , cat , target )
    facet.add_legend()

def plot_correlation_map( df ):
    corr = train.corr()
    ht , ax = plt.subplots( figsize =( 12 , 10 ) )
    cmap = sns.diverging_palette( 220 , 10 , as_cmap = True )
    ht = sns.heatmap(
        corr, 
        cmap = cmap,
        square=True, 
        cbar_kws={ 'shrink' : .9 }, 
        ax=ax, 
        annot = True, 
        annot_kws = { 'fontsize' : 12 })

def describe_more( df ):
    var = [] ; l = [] ; t = []
    for x in df:
        var.append( x )
        l.append( len( pd.value_counts( df[ x ] ) ) )
        t.append( df[ x ].dtypes )
    #print(var,l,t)
    levels = pd.DataFrame( { 'Variable' : var , 'Levels' : l , 'Datatype' : t } )
    levels.sort_values( by = 'Levels' , inplace = True )
    return levels

train=pd.read_csv('train.csv')
test=pd.read_csv('test.csv')
full=train.append(test,ignore_index=True)
full.head()

train.describe()
describe_more(train)
describe_more(test)

train.info()
train.describe(include=['O'])
test.info()

####Studying Relationship between Different Features and Survival, if found any relationship then we will select them for modelling.

plot_correlation_map(train)

train[['Pclass','Survived']].groupby(['Pclass'],  
as_index=False).mean().sort_values(by='Survived',ascending=False)

train[['Sex','Survived']].groupby(['Sex'], as_index=False).mean().sort_values(by='Survived',ascending=False)

train[['Parch','Survived']].groupby(['Parch'], as_index=False).mean().sort_values(by='Survived',ascending=False)

train[['SibSp','Survived']].groupby(['SibSp'], as_index=False).mean().sort_values(by='Survived',ascending=False)

g=sns.FacetGrid(train,col='Survived',row='Pclass')
g.map(plt.hist,'Age',bins=20)

grid=sns.FacetGrid(train,row='Embarked',size=2.2,aspect=1.6)
grid.map(sns.pointplot, 'Pclass','Survived','Sex',palette='deep')
grid.add_legend()

grid=sns.FacetGrid(train,row='Embarked',col='Survived',size=2.2,aspect=1.6)
grid.map(sns.barplot, 'Sex','Fare','Pclass',palette='deep')
grid.add_legend()

plot_distribution(train,var='Age',target='Survived',row='Sex')
plot_distribution(train,var='Fare',target='Survived',row='Sex')

plot_categories(train,cat='Embarked',target='Survived')
plot_categories(train,cat='Pclass',target='Survived')
plot_categories(train,cat='Sex',target='Survived')
plot_categories(train,cat='SibSp',target='Survived')
plot_categories(train,cat='Parch',target='Survived')

####Creating features


Sex=full['Sex'].map({'female':1,'male':0}).astype(int)
#Sex.head()

Emb=full.Embarked.fillna(full.Embarked.dropna().mode())
Embarked=pd.get_dummies(Emb,prefix='Embarked',columns=['Embarked'])
#Embarked.head()
Pclass=pd.get_dummies(full.Pclass,prefix='Pclass')
#Pclass.head()

#Title=pd.DataFrame()
#Title['Title']=full['Name'].map(lambda name:name.split(',')[1].split('.')[0].strip())
#Title.head()
Title=full['Name'].str.extract('([A-Za-z]+)\.',expand=False)
#Title.head()
pd.crosstab(Title,Sex)
Title=Title.replace(['Lady', 'Countess','Don', 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
Title=Title.replace({'Col':'Officer','Capt':'Officer','Dr':'Officer','Mlle':'Miss','Ms':'Miss','Mme':'Mrs','Major':'Officer'})
full['Titlee']=Title
#full.head()
plot_categories(full.loc[:891,:],cat='Titlee',target='Survived')
Title=pd.get_dummies(Title,columns=['Title'])
Title.head()


Cabin = full.Cabin.fillna( 'U' )
Cabin = Cabin.map( lambda c : c[0] )
full['CabinT']=Cabin
plot_categories(full.loc[0:890],cat='CabinT',target='Survived')
pd.value_counts(full.CabinT[0:890])
Cabin = pd.get_dummies(Cabin, prefix = 'Cabin' )
Cabin.head()


Family=pd.DataFrame()
Family['FamilySize']=(full['Parch']+full['SibSp'])+1
full['FamilySize']=Family

fig,ax=plt.subplots()
ax.hist(full.FamilySize[0:890],bins=11)
ax.set_xlabel('FamilySize')
ax.set_ylabel('Frequency')
fig,ax=plt.subplots()
sns.barplot(x='FamilySize',y='Survived',data=full.loc[0:890,:],ax=ax)
#pd.value_counts(Family.FamilySize[0:890])
#FamilySize=pd.get_dummies(Family['FamilySize'],prefix='FamilySize')
#or
Family['Alone']=Family['FamilySize'].map(lambda s:1 if s==1 else 0)
Family['Family_Small']=Family['FamilySize'].map(lambda s:1 if 2<=s<=4 else 0)
Family['Family_Large']=Family['FamilySize'].map(lambda s:1 if s>=5 else 0)
Family.head()


#full.Ticket.loc[0:15]
def cleanTicket( ticket ):
    ticket = ticket.replace( '.' , '' )
    ticket = ticket.replace( '/' , '' )
    ticket = ticket.split()
    ticket = map( lambda t : t.strip() , ticket )
    ticket = list(filter( lambda t : not t.isdigit() , ticket ))
    if len( ticket ) > 0:
        return ticket[0]
    else: 
        return 'XXX'
ticket = pd.DataFrame()
# Extracting dummy variables from tickets:
ticket[ 'Ticket' ] = full[ 'Ticket' ].map( cleanTicket )
full['Tickinfo']=ticket
fig,ax=plt.subplots()
sns.barplot(x='Tickinfo',y='Survived',data=full.loc[0:890,:],ax=ax)

ticket = pd.get_dummies( ticket[ 'Ticket' ] , prefix = 'Ticket' )
#ticket.head()
#tick=[val.replace('.','')for val in full[ 'Ticket' ].loc[0:3]]
#tick
grid=sns.FacetGrid(full,row='Sex',col='Pclass')
grid.map(plt.hist,'Age')

###filling missing Age values first
full['Sex']=Sex
#full.Sex.head()
guess_ages=np.zeros((2,3))
for i in range(2):
    for j in range(3):
        guess_ages[i,j]=full[(full.Sex==i) & (full.Pclass==j+1)&(full.Age.notna())]['Age'].mean()
guess_ages=guess_ages.astype(int)
#guess_ages
for i in range(2):
    for j in range(3):
        full.loc[(full.Sex==i) & (full.Pclass==j+1)& (full.Age.isna()),'Age']=(guess_ages[i,j])

#full.info()
Fare=full.Fare.fillna(full.Fare.mean())
full['Fare']=Fare
#Creating AgeBand
Train=pd.DataFrame()
Train=full.loc[0:890,:][['Age','Fare','Survived']]
Train['AgeBand']=pd.cut(Train['Age'],5)
Train[['AgeBand','Survived']].groupby(['AgeBand'],as_index=False).mean().sort_values(by='AgeBand',ascending=True)

full['AgeBand']=pd.cut(full['Age'],5)
AgeBand=pd.get_dummies(full['AgeBand'])
#AgeBand.head()

###Creating FareBand
grid=sns.FacetGrid(train,row='Survived')
grid.map(plt.hist,'Fare',bins=20)
fig,ax=plt.subplots()
sns.barplot(x='Survived',y='Fare',data=train,ax=ax)

Train['FareBand']=pd.qcut(Train['Fare'],4)
Train[['FareBand','Survived']].groupby(['FareBand'],as_index=False).mean().sort_values(by='FareBand',ascending=True)

full['FareBand']=pd.qcut(full['Fare'],4)
FareBand=pd.get_dummies(full['FareBand'])
#FareBand.head()


###Creating Training Set with the features created 
full_X=pd.concat([Sex,Family,AgeBand,FareBand,ticket,Cabin,Embarked,Title,Pclass],axis=1)
#full_X.head()
full_X.shape
#full_X.columns.values
train_valid_X = full_X[ 0:891 ]
#train_X.head
train_valid_y = train.Survived

test_X = full_X[ 891: ]
test_X.shape

###Splitting into training set and validation set
train_X, valid_X, train_y, valid_y= train_test_split( train_valid_X , train_valid_y , train_size = 0.7 )

###Trying different models using training set and checking accuracy on validation set
##Chose any model and run to predict the score

#model = GaussianNB()   ###poor score
#model=MLPClassifier()#solver='lbfgs',hidden_layer_sizes=(20,8),activation='logistic')
#model = KNeighborsClassifier(n_neighbors = 9)
#model = SVC(C=30,gamma=0.01)  ### Amongst high scorer
#model = LogisticRegression(C=0.1)
#model=DecisionTreeClassifier()
model = RandomForestClassifier(n_estimators=100)
model.fit( train_X , train_y )
print (model.score( train_X , train_y )*100,'%' , model.score( valid_X , valid_y )*100,'%')

###In next update Kfold Crossvalidatioin will be added to select the best model and checking for top features
