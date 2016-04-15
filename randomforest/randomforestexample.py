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