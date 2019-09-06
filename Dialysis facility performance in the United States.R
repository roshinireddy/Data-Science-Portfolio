# Building boruta algorithm

library(Boruta)
boruta.train_data<-Boruta(Org_Type~.-Total_Performance_Score,data=F1,doTrace=2)

plot(boruta.train_data, xlab = "", xaxt = "n")
lz<-lapply(1:ncol(boruta.train_data$ImpHistory),function(i)
  boruta.train_data$ImpHistory[is.finite(boruta.train_data$ImpHistory[,i]),i])
names(lz) <- colnames(boruta.train_data$ImpHistory)
Labels <- sort(sapply(lz,median))
axis(side = 1,las=2,labels = names(Labels),
     at = 1:ncol(boruta.train_data$ImpHistory), cex.axis = 0.7)

final.boruta <- TentativeRoughFix(boruta.train_data)
Print (final.boruta)


# random forest

library(caret)
library(randomForest)
set.seed(123)
control <- rfeControl(functions=rfFuncs, method="cv", number=10)
rfe.train <- randomForest(projectdata[,2:26],projectdata[,26], sizes=1:26, rfeControl=control)
projectdata$Performance<-as.factor(projectdata$Performance)
table(projectdata$Performance)
set.seed(200)
ind<-sample(2,nrow(projectdata),replace=TRUE,prob=c(0.7,0.3))
train<-projectdata[ind==1,]
test<-projectdata[ind==2,]
set.seed(222)
rf<-randomForest(Performance~.,
                 data=train,
                 ntree=300,
                 mtry =6,
                 importance = TRUE,
                 proximity = TRUE)
print(rf)

# confusion matrix for random forest

library(caret)
p1<-predict(rf,train)
confusionMatrix(p1,train$Performance)

p2<-predict(rf,test)
confusionMatrix(p2,test$Performance)

plot(rf)

# TUNING OF RANDOM FOREST

T<-tuneRF(train[,-26],train[,26],
          stepFactor = 0.01,
          plot = TRUE,
          ntreeTry = 500,
          trace = TRUE,
          improve = 0.0005)
print(T)

#number of nodes of the tree

hist(treesize(rf),
     main = "No of Nodes for the Trees",
     col = "green")

#feature extraction

varImpPlot(rf)
importance(rf)
varUsed(rf)

# partial dependance plot

partialPlot(rf,train,Total_Performance_Score,"1")
partialPlot(rf,train,Total_Performance_Score,"2")
partialPlot(rf,train,Total_Performance_Score,"3")
partialPlot(rf,train,Total_Performance_Score,"4")
partialPlot(rf,train,Total_Performance_Score,"5")
partialPlot(rf,train,Total_Performance_Score,"6")

#Linear regression
Call: lm(formula = Total_Performance_Score ~ ., data = logreg_data)

#logistics regression
#Iteration 1
model <- glm(Profit_org ~.,family=binomial(link='logit'),data=logreg_data_train)
summary(model)

#Iteration 2: -
model2 <-   glm(formula = Profit_org ~ No_of_Dialysis_STN + Percent_Patients_Serum_Phosphorus3.5.4.5mg.dl + 
                  Mortality_Rate_Facility + Readmission_Rate_Facility + Hospitalization_Rate_Facility, 
                family = binomial(link = "logit"), data = logreg_data_train)
summary(model2)

#Prediction of accuracy

fitted.results <- predict(model,logreg_data_test ,type='response')
fitted.results <- ifelse(fitted.results > 0.5,1,0)
misClasificError <- mean(fitted.results != logreg_data_test$Profit_org)
print(paste('Accuracy',1-misClasificError))

#neural networks


install.packages("neuralnet")
library(neuralnet)
NN_data<- read.csv(file.choose(), header = TRUE)
NN_data<- NN_data[,c(2,4,13,17,18,19,25)]
attach(NN_data)
NN_data$Profit_org[Org_Type == "Profit"] <- 1
NN_data$Profit_org[Org_Type == "Non-Profit"] <- 0
detach(NN_data)
nn<- neuralnet(Profit_org ~ No_of_Dialysis_STN + Percent_Patients_Serum_Phosphorus3.5.4.5mg.dl+
                 Mortality_Rate_Facility + Hospitalization_Rate_Facility+Readmission_Rate_Facility+
                 Total_Performance_Score, data = NN_data, hidden = 2, 
               err.fct = "ce", linear.output = FALSE)
nn
plot(nn)
nn$net.result
nn$weights
cov<-nn$covariate
nn$result.matrix
View(cov)
nn$net.result[[1]]
NN_data$Profit_org
nn1 = ifelse(nn$net.result[[1]]>0.5,1,0)
nn1
misclasserror<- mean(NN_data$Profit_org != nn1)
misclasserror

#Support vector machine
library(e1071)
svmmodel<-svm(Org_Type~., data = feature_data)
summary(svmmodel)
plot(svmmodel,data=feature_data,
     No_of_Dialysis_STN~Total_Performance_Score,)

#confusion matrix and Misclassification error 1: -
  
predsvm<-predict(svmmodel,feature_data)
tabsvm<-table(predicted=predsvm, Actual=feature_data$Org_Type)
tabsvm
1-sum(diag(tabsvm))/sum(tabsvm)
svmmodel2<-svm(Org_Type~., data = feature_data,kernel="linear")
summary(svmmodel2)
plot(svmmodel2,data=feature_data,
     No_of_Dialysis_STN~Total_Performance_Score,)

#Confusion matrix and Misclassification error 2: -
  
predsvm2<-predict(svmmodel2,feature_data)
tabsvm2<-table(predicted=predsvm2, Actual=feature_data$Org_Type)
tabsvm2
1-sum(diag(tabsvm2))/sum(tabsvm2)
svmmodel3<-svm(Org_Type~., data = feature_data,kernel="polynomial")
summary(svmmodel3)
plot(svmmodel3,data=feature_data,
     No_of_Dialysis_STN~Total_Performance_Score,)

#confusion matrix and Misclassification error 3

predsvm3<-predict(svmmodel3,feature_data)
tabsvm3<-table(predicted=predsvm3, Actual=feature_data$Org_Type)
tabsvm3
1-sum(diag(tabsvm3))/sum(tabsvm3)
svmmodel4<-svm(Org_Type~., data = feature_data,kernel="sigmoid")
summary(svmmodel4)
plot(svmmodel4,data=feature_data,
     No_of_Dialysis_STN~Total_Performance_Score,)

#confusion matrix and Misclassification error 4

predsvm4<-predict(svmmodel4,feature_data)
tabsvm4<-table(predicted=predsvm4, Actual=feature_data$Org_Type)
tabsvm4
1-sum(diag(tabsvm4))/sum(tabsvm4)

#Hyperparameter optimization which helps to select the best model

set.seed(123)
tmodel<-tune(svm,Org_Type~.,data = feature_data,
             kernel="linear",ranges = list(cost=c(0.001,0.01,0.1,1,10)))
plot(tmodel)
summary(tmodel)

#Best model from SVM
  
groupAbestmodel<-tmodel$best.model
summary(groupAbestmodel)
