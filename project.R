library(caret)

if(!file.exists("pml-training.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", "pml-training.csv")
}
if(!file.exists("pml-testing.csv")){
  download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", "pml-testing.csv")
}

training<-read.csv("pml-training.csv")
predictionData<-read.csv("pml-testing.csv")

noTestData<-colSums(!is.na(predictionData))==nrow(predictionData)
smallTraining<-training[,noTestData]
smallPredictionData<-predictionData[,noTestData]

zero<-nearZeroVar(smallTraining)

smallTraining<-smallTraining[,c(-1, -2, -3, -4, -5, -6, -7)]
smallPredictionData<-smallPredictionData[,c(-1, -2, -3, -4, -5, -6, -7)]

set.seed(1234)
inTrain=createDataPartition(smallTraining$classe, p=0.6, list=F)

trControl<-trainControl(method="cv", number=5, allowParallel=T)
m1<-train(classe~., method="rf", data=smallTraining[inTrain,], trControl=trControl)
m2<-train(classe~., method="gbm", data=smallTraining[inTrain,], trControl=trControl)

c1=confusionMatrix(predict(m1, newdata=smallTraining[-inTrain,]), smallTraining[-inTrain,]$classe)
c2=confusionMatrix(predict(m2, newdata=smallTraining[-inTrain,]), smallTraining[-inTrain,]$classe)

v1<-varImp(m1)$importance
v2<-varImp(m2)$importance

print(c1)
print(c2)

prePlotData<-cbind(rf=v1, gbm=v2)
colnames(prePlotData)=c("rf", "gbm")
plotData<-melt(as.matrix(prePlotData), id.vars=c(1, 2))
colnames(plotData)<-c("variableName", "method", "value")
p<-ggplot(plotData,aes(x = variableName,y = value)) + geom_bar(aes(fill = method),position = "dodge", stat="identity")+coord_flip()+facet_grid(~method)+scale_x_discrete(limits=rownames(v1)[order(v1, decreasing=F)])+labs(x="Variable importance", y="Variable name", title="Variable importance")
print(p)
