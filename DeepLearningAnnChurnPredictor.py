#importing libraries
import numpy as np
import pandas as pd
import tensorflow
import matplotlib.pyplot as plt
import seaborn as sns #Python data visualization library based on matplotlib.
pd.options.display.max_rows = None
pd.options.display.max_columns = None

df = pd.read_csv(r"\bank_churn.csv")
feature_names = ['customer_id', 'credit_score','age','tenure','balance','products_number','credit_card','active_member','estimated_salary']

print("size of the dataset:")
print(df.shape) #to get the matrix size
print("\n")
print("Columns          Missing Values")
print(df.isnull().sum())  #checks col list and sees is any values are missing
print("\n")
print("Unique count of variables:")
print(" ")
print(df.nunique())
print(" ")
print(df.head())

#data preprocessing

X = df[feature_names] #train
y = df["churn"] #test

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)
# Feature Scaling (very important)
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#creating the artifical neural network
import keras
import sys
from keras.models import Sequential #to initialize NN
from keras.layers import Dense #used to create layers in NN

#Create a sequential ANN then construct complete ANN by adding layers
classifier = Sequential()

#Input Layer
classifier.add(Dense(units = 6, kernel_initializer = 'uniform', activation = 'relu', input_dim = 9 ))
#forward propagation

#adding a hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform'))

#adding another hidden layer
classifier.add(Dense(activation = 'relu', units=6, kernel_initializer='uniform'))

#adding the output layer
classifier.add(Dense(activation = 'sigmoid', units=1, kernel_initializer='uniform'))

classifier.compile(optimizer='adam', loss = 'binary_crossentropy', metrics=['accuracy'])

classifier.fit(X_train, y_train, batch_size = 10, epochs = 100)
print("\n")
print(classifier)
print("prediction values")
y_pred = classifier.predict(X_test)

print(y_pred)
print("\n")
print("confusion matrix")

from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred.round())
print(cm)
cm_df = pd.DataFrame(cm,index=["No Churn","Churn"],columns=["No Churn","Churn"])
print("\n")


#visually representing a box plot
plt.figure(figsize=(8,6))
sns.heatmap(cm_df,annot=True ,cmap="Blues",fmt="g" )
plt.title("Confusion Matrix",y=1.1)
plt.ylabel("Actual Label")
plt.xlabel("Predicted Label")
plt.show()

print(((cm[0][0]+cm[1][1])*100)/(cm[0][0]+cm[1][1]+cm[0][1]+cm[1][0]), '% overall ACCURACY of trained model on the dataset')
print("\n")

#plotting the roc curve

from sklearn.metrics import roc_curve, auc

fpr, tpr, threshold = roc_curve(y_test, y_pred)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, label = 'P AUC = %0.2f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()

#plotting a pie chart

labels = 'Exited', 'Retained'
sizes = [df.churn[df['churn']==1].count(), df.churn[df['churn']==0].count()]
explode = (0, 0.1)
fig1, ax1 = plt.subplots(figsize=(10, 8))
ax1.pie(sizes, explode=explode, labels=labels, autopct='%1.1f%%',shadow=True, startangle=90)
ax1.axis('equal')
plt.title("Proportion of customer churned and retained", size = 20)
plt.show()

# plotting counter plots

_, ax = plt.subplots(1, 3, figsize=(18, 6))
plt.subplots_adjust(wspace=0.3)
sns.countplot(x = "products_number", hue="churn", data = df, ax= ax[0])
sns.countplot(x = "credit_card", hue ="churn", data = df, ax = ax[1])
sns.countplot(x = "active_member", hue="churn", data = df, ax = ax[2])

fig, axarr = plt.subplots(2, 2, figsize=(20, 12))
sns.countplot(x='country', hue = 'churn',data = df, ax=axarr[0][0])
sns.countplot(x='gender', hue = 'churn',data = df, ax=axarr[0][1])
sns.countplot(x='credit_card', hue = 'churn',data = df, ax=axarr[1][0])
sns.countplot(x='active_member', hue = 'churn',data = df, ax=axarr[1][1])

#plotting scatter plots

_, ax = plt.subplots(1, 2, figsize=(15, 6))
sns.scatterplot(x = "balance", y = "age", data = df, hue="churn", ax = ax[0])
sns.scatterplot(x = "balance", y = "credit_score", data = df, hue="churn", ax = ax[1])

#plotting a heat map

plt.subplots(figsize=(11,8))
sns.heatmap(df.corr(), annot=True, cmap="RdYlBu")
plt.show()

#plotting a KDE PLOT which is used to  for visualizing the Probability Density of a continuous variable

facet = sns.FacetGrid(df, hue="churn",aspect=3)
facet.map(sns.kdeplot,"balance",shade= True)
facet.set(xlim=(0, df["balance"].max()))
facet.add_legend()
plt.show()

#plotting a swarmplot
plt.figure(figsize=(8, 8))
sns.swarmplot(x = "credit_card", y = "age", data = df, hue="churn")

#plotting box plots
fig, axarr = plt.subplots(3, 2, figsize=(20, 12))
sns.boxplot(y='credit_score',x = 'churn', hue = 'churn',data = df, ax=axarr[0][0])
sns.boxplot(y='age',x = 'churn', hue = 'churn',data = df , ax=axarr[0][1])
sns.boxplot(y='tenure',x = 'churn', hue = 'churn',data = df, ax=axarr[1][0])
sns.boxplot(y='balance',x = 'churn', hue = 'churn',data = df, ax=axarr[1][1])
sns.boxplot(y='products_number',x = 'churn', hue = 'churn',data = df, ax=axarr[2][0])
sns.boxplot(y='estimated_salary',x = 'churn', hue = 'churn',data = df, ax=axarr[2][1])


