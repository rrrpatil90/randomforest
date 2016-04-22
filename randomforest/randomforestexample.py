import pandas as pd
import numpy as np
from difflib import SequenceMatcher
from sklearn.ensemble import RandomForestClassifier
import jellyfish as jf
import re, math
import os, sys
import cPickle
from pyspark.sql import SQLContext
#from pyspark.sql.types import *
from pyspark import SparkContext
import pandas.tools.util as tools
from pandas.core.index import MultiIndex

## take parameters
#inputFile = "E:\DMLE Project\Madhu\data\HierarchyInputWithBFMResult_brandFamily.csv"
#targetFile = "E:\DMLE Project\Madhu\data\\target_brandFamily.csv"
inputFile = sys.argv[1]
targetFile = sys.argv[2]
modelPath = "E:\DMLE Project\Madhu\ModelRepo"
Client = "Nestle"
Category = "Water"

## define function to calculate jaccard similarity
def jaccard_similarity(x, y):
    intersection_cardinality = len(set.intersection(*[set(x), set(y)]))
    union_cardinality = len(set.union(*[set(x), set(y)]))
    return intersection_cardinality / float(union_cardinality)

def rf(inputLevelWithBFM,targetLevelName):
    #inputLevelWithBFM = pd.read_csv(inputFile, header=0)
    # inputLevelWithBFM = pd.read_csv('data/HierarchyInputWithBFMResult_brandFamily.csv', header=0)
    ## Read input datasets - input labels and target labels
    
    #targetLevelName = pd.read_csv(targetFile, header=0)
    # targetLevelName = pd.read_csv('data/target_brandFamily.csv', header=0)
    targetLevelName = pd.DataFrame({'targetLevel': pd.unique(targetLevelName.targetLevel)})
    ## Convert the labels as character string
    #inputLevelWithBFM.astype(str)
    
    ## Take top 200 for test
    inputLevelWithBFM = inputLevelWithBFM[:50]
    ##########################################################################################
    ###### Predicting 'All Other' versus Non-All Other
    ##########################################################################################
    
    
    ## Define Train and Test data. Train data holds randomly chosen 70% records
    ## Rest of 30% records are considered as test data
    np.random.seed([101])
    trainData = inputLevelWithBFM.sample(frac=.7, replace=False)
    ## Store the data for prediction
    predDataAllOther = inputLevelWithBFM
    
    ################# Prepare TRAIN data #####
    
    ## Train Data - Pre-processing
    inputLabel_df = pd.DataFrame(trainData.loc[:, 'inputLabel'])
    # labelCompare = list(trainData['inputLabel'].str.replace('[^0-9A-Za-z]', ''))
    inputLabel_df.loc[:, 'key'] = 0
    
    ## Make tuple of input labels for which longest common substring needs
    ## to be identified
    labelCompare = pd.merge(inputLabel_df, inputLabel_df, how='outer', on='key').drop(['key'], axis=1)
    labelCompare.columns = ['inputLabel', 'inputLabel_compare']
    labelCompare = labelCompare.drop(labelCompare[labelCompare['inputLabel'] == labelCompare['inputLabel_compare']].index)
    labelCompare.loc[:, 'inputNoSplChr'] = labelCompare.loc[:, 'inputLabel'].str.replace('[^0-9A-Za-z]', '')
    labelCompare.loc[:, 'inputCompareNoSplChr'] = labelCompare.loc[:, 'inputLabel_compare'].str.replace('[^0-9A-Za-z]', '')
    ## Prepare an empty data frame to store the longest common substring
    ## corresponding to each input label
    ## Check whether any number is present in input label
    ## Determine the length of input lable
    labelWithLCS = pd.DataFrame()
    labelCompareLength = len(labelCompare)
    # def getLongestCommonSubstring(x, y):
    #    s = SequenceMatcher(None, x, y)
    #    return s.find_longest_match(0, len(x), 0, len(y)).size
    
    ## Store the longest common substring, flag to determine whether a number is present and
    ## the length for each input label
    for i in range(0, (labelCompareLength)):
        a = pd.DataFrame(labelCompare.iloc[i, [2]])
        a = pd.DataFrame.to_string(a, header=False, index=False)[1:]
    
        b = pd.DataFrame(labelCompare.iloc[i, [3]])
        b = pd.DataFrame.to_string(b, header=False, index=False)[1:]
    
        c = list(labelCompare.iloc[i, [0]])
        # c = pd.DataFrame.to_string(c, header=False, index=False)
        # m = longest_common_substring(a,b)
        s = SequenceMatcher(None, a, b)
        result = s.find_longest_match(0, len(a), 0, len(b))
        if result.size > 3:
            lcs = a[(result.a):(result.a + result.size)]
            labelWithLCS_tmp = pd.DataFrame({'longestCommonString': lcs, 'inputLabel': c})
            labelWithLCS = labelWithLCS.append(labelWithLCS_tmp, ignore_index=True)
    ## Create dummy variables for all longest common substring categorical variables
    ## Remove 'longestCommonString' string from the dummy variable names
    
    dummies = pd.get_dummies(labelWithLCS.loc[:, 'longestCommonString'])
    labelWithLCS_withdummy = pd.concat([labelWithLCS, dummies], axis=1)
    labelWithLCS_withdummy = labelWithLCS_withdummy.drop(['longestCommonString'], axis=1)
    labelWithLCSVariable = labelWithLCS_withdummy.groupby('inputLabel').max().reset_index()
    
    ## Prepare the train data for random forest model
    ## Define the class flag for 'ALL OTHER' and 'ALL_OTHER'
    
    trainDataWithLCS = pd.merge(labelWithLCSVariable, trainData, on='inputLabel', how='inner')
    trainDataWithLCS['inputLength'] = map(len, trainDataWithLCS['inputLabel'])
    trainDataWithLCS['numPresence'] = trainDataWithLCS['inputLabel'].str.contains('\d', na=False).astype(float)
    trainDataWithLCS['splChrPresence'] = trainDataWithLCS['inputLabel'].str.replace(' ', '').str.contains('[^0-9A-Za-z]',
                                                                                                          na=False).astype(
        float)
    trainDataWithLCS.loc[:, 'AllOtherFlag'] = trainDataWithLCS.loc[:, 'manualMappedLabel'].apply(
        lambda x: 1 if x in ["ALL OTHER", "ALL_OTHER"] else 0)
    trainDataWithLCS = trainDataWithLCS.drop(['inputLabel', 'bfmMappedLabel', 'manualMappedLabel'], axis=1)
    
    ## Fit a random forest model using down sampling techniques for unbalanced data
    ## Check the class for low frequency and use them for sample size
    features1 = trainDataWithLCS.columns[:-1]
    y, _ = pd.factorize(trainDataWithLCS.loc[:, 'AllOtherFlag'], sort=True)
    lowClassFreq = len(trainDataWithLCS) - trainDataWithLCS['AllOtherFlag'].sum(axis=0)
    forest = RandomForestClassifier(n_estimators=1000, class_weight="balanced")
    forest.fit(trainDataWithLCS[features1], y)
    
    
    ############### Prepare TEST data / Prediction data
    
    ## Remove special characters from input labels
    predDataAllOther.loc[:, 'inputNoSplChr'] = predDataAllOther.loc[:, 'inputLabel'].str.replace('[^0-9A-Za-z]', '')
    
    ## Define the class variable and make it as categorical/factor
    predDataAllOther.loc[:, 'AllOtherFlag'] = predDataAllOther.loc[:, 'manualMappedLabel'].apply(
        lambda x: 1 if x in ["ALL OTHER", "ALL_OTHER"] else 0)
    predDataAllOther.loc[:, 'AllOtherFlag'], _ = pd.factorize(predDataAllOther.loc[:, 'AllOtherFlag'], sort=True)
    
    ## Define longest common string dummies
    predDataLength = len(predDataAllOther)
    impVars = features1
    for i in range(0, len(impVars)):
        dummyx = str(impVars[i])
        predDataAllOther[dummyx] = predDataAllOther['inputNoSplChr'].str.contains(dummyx, na=False).astype(float)
    
    ## Define input length, number presence flag in input, special character
    ## presence flag in input
    predDataAllOther['inputLength'] = map(len, predDataAllOther['inputLabel'])
    predDataAllOther['numPresence'] = predDataAllOther['inputLabel'].str.contains('\d', na=False).astype(float)
    predDataAllOther['splChrPresence'] = predDataAllOther['inputLabel'].str.replace(' ', '').str.contains('[^0-9A-Za-z]',
                                                                                                          na=False).astype(
        float)
    
    ## Predict the class using the trained random forest model
    preds = forest.predict(predDataAllOther[impVars])
    predDataAllOther['predAllOtherFlag'] = preds
    predDataAllOther['shConfidence'] = 1 - forest.predict_proba(predDataAllOther[impVars])[:, 0]
    
    ## Define the mapped label as 'ALL OTHER' if the prediction class is 1
    predDataAllOther['shMappedLabel'] = predDataAllOther['predAllOtherFlag'].apply(
        lambda x: 'All OTHER' if x == 1 else 'Non All Other')
    
    ## Validate the results using confusion matrix
    pd.crosstab(predDataAllOther['AllOtherFlag'], preds, rownames=['actual'], colnames=['preds'])
    
    ## Save the predicted results
    
    
    ##########################################################################################
    ###### Predicting labels for all Non-All Other
    ##########################################################################################
    
    ## Extract data for only non-all other labels
    # inputLevelNAOWithBFM = subset(inputLevelWithBFM, !(inputLevelWithBFM$manualMappedLabel %in% c("ALL OTHER", "ALL_OTHER") ) )
    
    inputLevelNAOWithBFM = predDataAllOther.query('predAllOtherFlag == 0')[
        ['manualMappedLabel', 'inputLabel', 'bfmMappedLabel']]
    
    ## Define Train and Test data. Train data holds randomly chosen 70% records.
    ## Rest of 30% records are considered as test data
    trainData = inputLevelNAOWithBFM.sample(frac=.7, replace=False).reset_index()
    # testData = inputLevelNAOWithBFM.drop(trainData.index)
    
    ##################################################################
    ##part2
    ##################################################################
    
    ## Extract data for only non-all other labels
    # inputLevelNAOWithBFM = subset(inputLevelWithBFM, !(inputLevelWithBFM$manualMappedLabel %in% c("ALL OTHER", "ALL_OTHER") ) )
    
    inputLevelNAOWithBFM = predDataAllOther.query('predAllOtherFlag == 0')[
        ['manualMappedLabel', 'inputLabel', 'bfmMappedLabel']]
    
    ## Define Train and Test data. Train data holds randomly chosen 70% records.
    ## Rest of 30% records are considered as test data
    trainData = inputLevelNAOWithBFM.sample(frac=.7, replace=False).reset_index()
    # testData = inputLevelNAOWithBFM.drop(trainData.index)
    
    ############## Prepare the TRAIN data
    
    ## Create variable as input labels without any special character
    trainData['inputNoSplChr'] = trainData['inputLabel'].str.replace('[^0-9A-Za-z]', '')
    trainData = trainData.reset_index(drop=True)
    targetLevelName['targetNoSplChr'] = targetLevelName['targetLevel'].str.replace('[^0-9A-Za-z]', '')
    
    ## Create the empty similarity matrix
    similarMatrix = pd.DataFrame()
    
    ## Update similarity matrix with all input labels and corresponding mapped labels
    ## having maximum similarity measures.
    ## The maximum similarity might return multiple mapped labels.
    ## Rules needs to be applied later to select the best mapped one
    ## Measure the similarity between the target label and the input label after removing the special characters
    lengthTrain = len(trainData)
    
    for i in range(0, lengthTrain):
        inputx = trainData.loc[i, 'inputNoSplChr']
        ## Measure Jaro-Winker similarity, equals to 1 - jw in R
        jarowinkerSim = targetLevelName['targetNoSplChr'].apply(lambda x: jf.jaro_winkler(unicode(inputx), unicode(x)))
        kmax = max(jarowinkerSim)
        targetmax = targetLevelName.loc[jarowinkerSim.idxmax(), 'targetNoSplChr']
        tempDF = pd.DataFrame({'targetNoSplChr': [targetmax]})
        tempDF['similarityScore'] = kmax
        tempDF['measures'] = 'JaroWinkler'
        tempDF['inputNoSplChr'] = inputx
        similarMatrix = similarMatrix.append(tempDF, ignore_index=True)
        ## Measure Damerau-Levenshtein similarity equals to dl in R
        dlavenshteinSim = targetLevelName['targetNoSplChr'].apply(
            lambda x: jf.damerau_levenshtein_distance(unicode(inputx), unicode(x))) / targetLevelName[
                              'targetNoSplChr'].apply(
            lambda x: max(len(inputx), len(x)))
        kmax = max(dlavenshteinSim)
        targetmax = targetLevelName.loc[dlavenshteinSim.idxmax(), 'targetNoSplChr']
        tempDF = pd.DataFrame({'targetNoSplChr': [targetmax]})
        tempDF['similarityScore'] = kmax
        tempDF['measures'] = 'DamerauLevenshtein'
        tempDF['inputNoSplChr'] = inputx
        similarMatrix = similarMatrix.append(tempDF, ignore_index=True)
        ## Measure Jaccard similarity equals to 1 - jaccard in R
        jaccardSim = targetLevelName['targetNoSplChr'].apply(lambda x: jaccard_similarity(inputx, x))
        kmax = max(dlavenshteinSim)
        targetmax = targetLevelName.loc[jaccardSim.idxmax(), 'targetNoSplChr']
        tempDF = pd.DataFrame({'targetNoSplChr': [targetmax]})
        tempDF['similarityScore'] = kmax
        tempDF['measures'] = 'Jaccard'
        tempDF['inputNoSplChr'] = inputx
        similarMatrix = similarMatrix.append(tempDF, ignore_index=True)
        ## Measure Cosine similarity
        # cosineSimMatrix =
        # kmax = max(dlavenshteinSim)
        # tempDF = pd.DataFrame()
        # tempDF['targetNoSplChr'] =
        # tempDF['similarityScore'] = kmax
        # tempDF['measures'] = 'Cosine'
        # tempDF['inputNoSplChr'] = inputx
        # similarMatrix = similarMatrix.append(tempDF, ignore_index=True)
        #
        # ## Measure Longest Common Substring similarity
        # lcsSimMatrix =
        # kmax = max(dlavenshteinSim)
        # tempDF = pd.DataFrame()
        # tempDF['targetNoSplChr'] =
        # tempDF['similarityScore'] = kmax
        # tempDF['measures'] = 'LongestCommonSubstr'
        # tempDF['inputNoSplChr'] = inputx
        # similarMatrix = similarMatrix.append(tempDF, ignore_index=True)
    
    similarMatrix = pd.merge(similarMatrix, targetLevelName, how='left', on='targetNoSplChr')
    
    similarMatrix = similarMatrix.rename(columns={'targetLevel': 'mappedLabel'})
    similarMatrix = similarMatrix.rename(columns={'targetNoSplChr': 'mappedNoSplChr'})
    
    ## Merge the similarity score with training data
    trainDataWithSimilarity = pd.merge(trainData, similarMatrix, how='inner', on='inputNoSplChr')
    
    ## Find the maximum length of the longest common string (without the special character) between the
    ## input label and mapped label, corresponding to each similarity score##?mapped=target?
    dataLength = len(trainDataWithSimilarity)
    
    for i in range(0, dataLength):
        a = trainDataWithSimilarity.loc[i, 'inputNoSplChr']
        b = trainDataWithSimilarity.loc[i, 'mappedNoSplChr']
        s = SequenceMatcher(None, a, b)
        result = s.find_longest_match(0, len(a), 0, len(b))
        trainDataWithSimilarity.loc[i, 'maxLengthOfLCS'] = result.size
    
    trainDataWithSimilarity['inputLength'] = map(len, trainDataWithSimilarity['inputNoSplChr'])
    trainDataWithSimilarity['lcsLengthRatio'] = trainDataWithSimilarity['maxLengthOfLCS'] / trainDataWithSimilarity[
        'inputLength']
    
    ## Create similarity measure dummy variables
    dummies = pd.get_dummies(trainDataWithSimilarity['measures'])
    trainDataWithSimilarity = pd.concat([trainDataWithSimilarity, dummies], axis=1)
    
    ## Define the class varibales and make it as categorical variables
    for i in range(0, len(trainDataWithSimilarity)):
        a = trainDataWithSimilarity.loc[i, 'mappedLabel'].replace(' ', '')
        b = trainDataWithSimilarity.loc[i, 'manualMappedLabel'].replace(' ', '')
        if a == b:
            trainDataWithSimilarity.loc[i, 'matchSimilarityClass'] = 1
        else:
            trainDataWithSimilarity.loc[i, 'matchSimilarityClass'] = 0
    ## Store data for measuring variable importance using random forest
    trainDataForVarImp = trainDataWithSimilarity.drop(['measures', 'bfmMappedLabel', 'manualMappedLabel', 'inputLabel',
                                                       'inputNoSplChr', 'mappedLabel', 'mappedNoSplChr', 'similarityScore',
                                                       'index'],
                                                      axis=1)
    
    ## Fit a random forest model for feature selection
    
    
    ## Train the random forest model using important vriables
    features2 = trainDataForVarImp.columns[:-1]
    y, _ = pd.factorize(trainDataWithSimilarity.loc[:, 'matchSimilarityClass'], sort=True)
    trainDataNAORF = RandomForestClassifier(n_estimators=500)
    trainDataNAORF.fit(trainDataForVarImp[features2], y)
    print trainDataNAORF
    
    with open(modelPath + '/' + Client + Category + 'model', 'wb') as f:
        cPickle.dump([features1, features2, forest, trainDataNAORF], f)



#if __name__ == "__main__":
def main():
    sc = SparkContext(appName="MLRandomForestTrain")
    sqlContext = SQLContext(sc)


# df = sqlContext.read.load('/user/cloudera/DMLESPARK/Monamidata/brandFamily_manual_mapped.csv', 
#                           format='com.databricks.spark.csv', 
#                           header='true', 
#                           inferSchema='true')   
########################HierarchyInputWithBFMResult_brandFamily from Hdfs. 
#'/user/hue/oozie/workspaces/DMLE_BFM_V1_MLRFTRAIN-Dev/lib/HierarchyInputWithBFMResult_brandFamily.csv'
    inputLevelWithBFMrdd = sc.textFile(inputFile)
    inputLevelWithBFMrdd = inputLevelWithBFMrdd.map(lambda line: line.split(","))
    header = inputLevelWithBFMrdd.first()
    inputLevelWithBFMrdd = inputLevelWithBFMrdd.filter(lambda line:line != header)
    sparkdf = inputLevelWithBFMrdd.toDF()
    df = sparkdf.toPandas()
    df.columns = header
    inputLevelWithBFM = df
    inputLevelWithBFM = pd.DataFrame(inputLevelWithBFM)
########################Read target File from Hdfs. 
#'/user/hue/oozie/workspaces/DMLE_BFM_V1_MLRFTRAIN-Dev/lib/brandFamily_target.csv'
    targetLevelNamerdd = sc.textFile(targetFile) 
    targetLevelNamerdd = targetLevelNamerdd.map(lambda line: line.split(","))
    header = targetLevelNamerdd.first()
    targetLevelNamerdd = targetLevelNamerdd.filter(lambda line:line != header)
    sparkdf = targetLevelNamerdd.toDF()
    df = sparkdf.toPandas()
    df.columns = header
    targetLevelName = df
    targetLevelName = pd.DataFrame(targetLevelName)
    
    rf(inputLevelWithBFM,targetLevelName)
    




'''
from pyspark.sql import SQLContext
from pyspark.sql.types import StructType,StructField,StringType
from pyspark import SparkContext
    
def randomForestData(sc, sqlContext):
    inputLevelWithBFMrdd = sc.textFile('E:\DMLE Project\pyspark_conversion\\brandFamily_manual_mapped.csv') 
    inputLevelWithBFMrdd = inputLevelWithBFMrdd.map(lambda line: line.split(","))
    header = inputLevelWithBFMrdd.first()
    fields = [StructField(field_name, StringType(), True) for field_name in header]
    schema = StructType(fields)
    inputLevelWithBFMrdd = inputLevelWithBFMrdd.filter(lambda line:line != header)
    sparkdf = sqlContext.createDataFrame(inputLevelWithBFMrdd,schema)
    inputLevelWithBFM = sparkdf

    (trainingData, testData) = inputLevelWithBFM.randomSplit([0.7, 0.3])
    print type(testData)
    predDataAllOther = inputLevelWithBFM
    print type(predDataAllOther)
    inputLabel_df = trainingData.select('inputLabel')
    inputLabel_dfCopy = inputLabel_df
    inputLabel_dfCopy = inputLabel_dfCopy.withColumnRenamed('inputLabel', 'inputLabel_Compare')
    labelCompare = inputLabel_df.join(inputLabel_dfCopy,(inputLabel_df.inputLabel != inputLabel_dfCopy.inputLabel_Compare))
    labelCompare.show()
    
    
#if __name__ == "__main__":
def main():   
    sc = SparkContext(appName="RandomForest")
    sqlContext = SQLContext(sc)
    randomForestData(sc, sqlContext)
'''