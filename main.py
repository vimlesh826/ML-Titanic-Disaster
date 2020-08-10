#import neccessary libraries
import pandas as panda
import math 
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import pickle

#read csv file from source
train = panda.read_csv("Dataset/train.csv")
test = panda.read_csv("Dataset/test.csv")
test_y = panda.DataFrame(data=test["PassengerId"])

train["Age"] = train["Age"].fillna(30)
train["Alone"] = train["SibSp"]+train["Parch"]#create  new column in dataframe containing total family numbers count
train["Alone"].loc[train["Alone"]>0] = "with family"
train["Alone"].loc[train["Alone"]==0] = "without family"
train["Embarked"] = train["Embarked"].fillna("S")
train["Fare"] = train["Fare"].fillna(0)
train["Age"] = train["Age"].apply(math.ceil)
train["Fare"] = train["Fare"].apply(math.ceil)

test["Age"] = test["Age"].fillna(30)
test["Alone"] = test["SibSp"]+test["Parch"]#create  new column in dataframe containing total family numbers count
test["Alone"].loc[test["Alone"]>0] = "with family"
test["Alone"].loc[test["Alone"]==0] = "without family"
test["Embarked"] = test["Embarked"].fillna("S")
test["Fare"] = test["Fare"].fillna(0)
test["Age"] = test["Age"].apply(math.ceil)
test["Fare"] = test["Fare"].apply(math.ceil)

#creating dummy varibales (output in form of 0 or 1)
dummy_pclass = panda.get_dummies(train["Pclass"])
dummy_sex = panda.get_dummies(train["Sex"])
dummy_pclass.columns=["class1","class2", "class3"]
dummy_alone = panda.get_dummies(train["Alone"])
dummy_Embarked = panda.get_dummies(train["Embarked"])
dummy_alone = panda.get_dummies(train["Alone"])
train = panda.concat([train,dummy_pclass,dummy_Embarked,dummy_sex,dummy_alone], axis=1)

t_pclass = panda.get_dummies(test["Pclass"])
t_sex = panda.get_dummies(test["Sex"])
t_pclass.columns=["class1","class2", "class3"]
t_alone = panda.get_dummies(test["Alone"])
t_Embarked = panda.get_dummies(test["Embarked"])
t_alone = panda.get_dummies(test["Alone"])
test = panda.concat([test,t_pclass,t_Embarked,t_sex,t_alone], axis=1)

y = train["Survived"]   #output to be trained with

train.drop(["PassengerId", "Pclass", "Name",  "SibSp", "Parch", "Ticket", "Cabin", "Embarked", "Sex", "Alone", "Survived","Q","female","without family"], axis=1, inplace=True)
test.drop(["PassengerId", "Pclass", "Name",  "SibSp", "Parch", "Ticket", "Cabin", "Embarked", "Sex", "Alone", "Q","female","without family"], axis=1, inplace=True)
X = train
#train_X, val_X, train_y, val_y = train_test_split(X,y,test_size=0.8)

test_X = test

decision = RandomForestClassifier(max_leaf_nodes=7)
decision.fit(X,y)   
decision_model = decision.predict(test_X)   #prediction 
#print( accuracy_score(val_y,decision_model)) #to calculate accuracy of model.

filename = "trained_model.sav"  #model name and location to save
pickle.dump(filename,open(filename,'wb'))   #Saves trained model to given location

test_y["Survived"] = decision_model     #Copy model output to output dataframe
test_y.to_csv("output.csv",index=False) #converts output dataframe to .csv file
