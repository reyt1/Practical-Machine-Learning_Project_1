#download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv',
#              'classeData.csv', method = 'curl')
#download.file('https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv',
#              'classeTestingData.csv', method = 'curl')

library(caret)
library(rattle)
library(e1071)
library(ggplot2)
set.seed(12345)

wtLiftData <- read.csv(file = 'classeData.csv')
wtLiftDataTest <- read.csv(file = 'classeTestingData.csv')

# Split data into training + validation and testing sets
inTrainCV <- createDataPartition(wtLiftData$classe, p=0.8, list=FALSE) 
trainingCVRaw <- wtLiftData[inTrainCV,]
testingRaw <- wtLiftData[-inTrainCV,]

#-------------------------------------------------------------------------------
# Clean up the dataset by removing columns with only NAs and/or empty values
#-------------------------------------------------------------------------------
# Find NA values and exclude
notNA <- sapply(trainingCVRaw,function(i){sum(is.na(i))/length(i)})<0.9
trainingCVNotNA <- trainingCVRaw[, notNA]
#testingRaw <- testingRaw[, notNA]

# Find empty values and exclude
emptyFeatures <- as.logical(apply(trainingCVNotNA, 2, function(x) any(grepl('#DIV/0!', x))))
trainingCVNotEmpty <- trainingCVNotNA[, !emptyFeatures]
#testingRaw <- testingRaw[, !emptyFeatures]

# Find near zero variation values and exclude
nsv <- nearZeroVar(trainingCVNotEmpty, saveMetrics=T)
trainingCVClean <- trainingCVNotEmpty[, !nsv$nzv]
#testingRaw <- testingRaw[, !nsv$nzv]

# Remove indexing data (timestamps) and user_name
idxCols <- grep('timestamp|user|X|num', names(trainingCVClean))
trainingCVClean <- trainingCVClean[, -idxCols]
#testingClean <- testingRaw[, -idxCols]

#-------------------------------------------------------------------------------
# Split training dataset into 10 folds to perform cross-validation
trainingFolds <- createFolds(trainingCVClean$classe, k=30, list=TRUE, returnTrain=TRUE)

accuracy <- numeric(length(trainingFolds))

for (i in seq_along(trainingFolds)) {
    svmFit <- svm(classe~., data = trainingCVClean[trainingFolds[[i]],], cost = i)
    predictions <- predict(svmFit, trainingCVClean[-trainingFolds[[i]],])
    confMat <- confusionMatrix(predictions, trainingCVClean[-trainingFolds[[i]],]$classe)
    accuracy[i] <- confMat$overall[1]
}

qplot(y=accuracy,x=seq_along(accuracy),geom=c('smooth','point'), xlab = 'Cost', ylab = 'Accuracy')

svmFinal <- svm(classe~., data = trainingCVClean, cost = 15)

predictTest <- predict(svmFinal, testingRaw)

confusionMatrix(predictTest, testingRaw$classe)
