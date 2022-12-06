import pandas as pd;
import math
import warnings
import matplotlib.pyplot as plt

#Only using because otherwise you will see a looping warning about a function pandas is removing in the future
def fxn():
    warnings.filterwarnings("ignore")
fxn()

#Change csv file location to correct location in order to ensure agent runs

df = pd.read_csv(r'C:\Users\ethan\OneDrive - University of Wollongong\2022\Semester 2\CSIT AI\Lab\Lab1\iris.csv')
df.columns = ['Sepal Length','Sepal Width', 'Petal Length', 'Petal Width', 'Classification']
#Display inital Dataframe
print(df)

#Splits initial dataframe randomly into train and test dataframes respectively
trainSet = df.sample(frac = 0.5)
testSet = df.drop(trainSet.index)
actualTestSet = testSet.copy()
trainSet.reset_index(inplace=True)
testSet.reset_index(inplace=True)
testSet.loc[:,'Classification']='Unknown'
print("Training Set")
print(trainSet)

print("Testing Set")
print(testSet)

#Defines dataframes for processing
results = pd.DataFrame(columns=['OrigIndex','k','TestClassification'])
errorRate = pd.DataFrame(columns=['k','#OfTests','#OfErrors','Error Rate'])
errorRate['k'] = range(1,len(trainSet),2)
errorRate.loc[:,'#OfTests'] = 0
errorRate.loc[:,'#OfErrors'] = 0
errorRate.loc[:,'Error Rate'] = 0

#Euc Distance calculating loop
for y in range(0,len(testSet)):
    distances = []
    for x in range(0,len(trainSet)):
        distances.append((math.sqrt((trainSet.iat[x,1] - testSet.iat[y,1])**2 + (trainSet.iat[x,2] - testSet.iat[y,2])**2 + (trainSet.iat[x,3] - testSet.iat[y,3])**2) + (trainSet.iat[x,4] - testSet.iat[y,4])**2,trainSet.iat[x,5]))
        #print(distances)
    eucDistance = pd.DataFrame(distances,columns=['eucDist','TrainClassification'])
    eucDistance.sort_values(by=['eucDist'],inplace=True)
    allClassif = []
    for k in range(1,len(trainSet),2):
        kCloseN = eucDistance.head(k)
        cls = kCloseN['TrainClassification'].mode()
        cls = cls.iloc[0]
        results = results.append({'OrigIndex' : testSet.iat[y,0], 'k' : k, 'TestClassification' : cls}, ignore_index=True)


#print(testSet)
#print(actualTestSet)
print("Results")
print(results)


for p in range(0,len(results)):
    currentRow = (results.loc[p,'k']-1)/2
    if results.loc[p,'TestClassification'] != actualTestSet.loc[results.loc[p,'OrigIndex'],'Classification']:
        errorRate.loc[currentRow,'#OfErrors'] = errorRate.loc[currentRow,'#OfErrors']+1
    errorRate.loc[currentRow,'#OfTests'] = errorRate.loc[currentRow,'#OfTests']+1
for o in range(0,len(errorRate)):
    errorRate.loc[o,'Error Rate'] = errorRate.loc[o,'#OfErrors']/errorRate.loc[o,'#OfTests']

print("Errors Identified")
print(errorRate)
errorRate.plot(y='Error Rate', x='k', kind='bar')
plt.show()
