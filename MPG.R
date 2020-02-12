# Seed for official run: seed(3)

# This R script explores decision tree classifcation models in order to classify automobiles as having a low or high MPG rating
# based on various automobile specifications

# Access needed libraries for classification analysis
library(randomForest)
library(tree)
library(gbm)

x = read.table(file.choose(),header=F) # Select auto-mpg data file

names(x) = c("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name") # Add column labels to data set

table(is.na(x)) # Check for missing values, 14 missing values

Auto = x[complete.cases(x),]  # Delete all missing values

attach(Auto) # Attach Auto data frame to R

Auto <- Auto[,-9]; # remove Car Name column

mpgClass = ifelse(mpg<=23,"Low","High") # Split continuous mpg (response) variable at 23 (closest integer value to both mean and median of mpg variable) to convert it to a class label

Auto=data.frame(Auto, mpgClass) # Merge binary mpgClass variable with original data set

Auto=Auto[,-1] # Remove original, continuous mpg variable

# Split full data set into training data set (2/3 = 262) and test data set
set.seed(3)
train=sample(1:nrow(Auto), 262)
Auto.train=Auto[train,]
Auto.test=Auto[-train,]
mpgClass.train=mpgClass[train]
mpgClass.test=mpgClass[-train]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Holdout Method----------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Generate a decision tree from the training data set
tree.Auto=tree(mpgClass~.,Auto.train)
plot(tree.Auto)
text(tree.Auto,pretty=0)

# Check error rates
tree.pred1=predict(tree.Auto,Auto.train,type="class") # Prediction on the training dataset
table(tree.pred1,mpgClass.train) # Confusion matrix to check accuracy
mean(tree.pred1!=mpgClass.train) # Training error rate = 0.04580153

tree.pred=predict(tree.Auto,Auto.test,type="class") # Prediction on the test dataset
table(tree.pred,mpgClass.test) # Confusion matrix to check accuracy
mean(tree.pred!=mpgClass.test) # Test error rate = 0.07692308

# Consider pruning the tree
cv.Auto=cv.tree(tree.Auto,FUN=prune.misclass)
cv.Auto

plot(cv.Auto$size ,cv.Auto$dev ,type="b") # Graph of error vs tree size

# Prune the tree to size 6
prune.Auto=prune.misclass(tree.Auto,best=6)
plot(prune.Auto)
text(prune.Auto,pretty=0)

# Check error rates for size 6 tree
tree.pred1=predict(prune.Auto,Auto.train,type="class")
table(tree.pred1,mpgClass.train)
mean(tree.pred1!=mpgClass.train) #training error = 0.04961832

tree.pred=predict(prune.Auto,Auto.test,type="class")
table(tree.pred,mpgClass.test)
mean(tree.pred!=mpgClass.test) #test error is 0.06923077

# Prune the tree to size 7
prune.Auto=prune.misclass(tree.Auto,best=7)
plot(prune.Auto)
text(prune.Auto,pretty=0)

# Check error rates for size 7 tree
tree.pred1=predict(prune.Auto,Auto.train,type="class")
table(tree.pred1,mpgClass.train)
mean(tree.pred1!=mpgClass.train) #training error = 0.04580153

tree.pred=predict(prune.Auto,Auto.test,type="class")
table(tree.pred,mpgClass.test)
mean(tree.pred!=mpgClass.test) #test error is 0.07692308

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Bagging Method----------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Note: There are 7 predictor variables

ResultBagging.test <- matrix(1,48)
colnames(ResultBagging.test) = "Test_Error_Rate"

for(i in seq(from=300,to=5000,by=100)){
  
  tree.Auto=randomForest(mpgClass~.,Auto.train, ntree=i,mtry=7)
  tree.pred=predict(tree.Auto,Auto.test,type="class")
  table(tree.pred,mpgClass.test)
  ResultLoop <- mean(tree.pred!=mpgClass.test)
  ResultBagging.test[(i-200)/100,] <- c(ResultLoop)
  
}

# Find the index of the test error error matrix with the lowest error
ResultBagging.test[which.min(ResultBagging.test),] # Best test error = 0.03846154

# Number of trees producing best test error
300*which.min(ResultBagging.test) # 600 trees

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Random Forest Method----------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

#Note: 2 < sqrt(7) < 3

#Random Forest with No. of variable = 2
ResultForest.test <- matrix(1,48)
colnames(ResultForest.test) = "Test_Error_Rate"

for(i in seq(from=300,to=5000,by=100)){
  
  tree.Auto=randomForest(mpgClass~.,Auto.train, ntree=i,mtry=2)
  tree.pred=predict(tree.Auto,Auto.test,type="class")
  table(tree.pred,mpgClass.test)
  ResultLoop <- mean(tree.pred!=mpgClass.test)
  ResultForest.test[(i-200)/100,] <- c(ResultLoop)
  
}

# Find the index of the test error error matrix with the lowest error
ResultForest.test[which.min(ResultForest.test),] # Best test error = 0.04615385

# Number of trees producing best test error
300*which.min(ResultForest.test) # 900 trees

#Random Forest with No. of variable = 3
for(i in seq(from=300,to=5000,by=100)){
  
  tree.Auto=randomForest(mpgClass~.,Auto.train, ntree=i,mtry=3)
  tree.pred=predict(tree.Auto,Auto.test,type="class")
  table(tree.pred,mpgClass.test)
  ResultLoop <-  mean(tree.pred!=mpgClass.test)
  ResultForest.test[(i-200)/100,] <- c(ResultLoop)
  
}

# Find the index of the test error error matrix with the lowest error
ResultForest.test[which.min(ResultForest.test),] # Best test error = 0.04615385

# Number of trees producing best test error
300*which.min(ResultForest.test) # 300 trees

importance(tree.Auto) # Check variable importance
varImpPlot(tree.Auto) # Displacement is most important variable

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Boosting Method---------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Convert the mpgClass variable from 2 levels of strings to binary {0="Low",1="High"}
Auto$mpgClass=ifelse(Auto$mpgClass=="High",1,0)
Auto.train=Auto[train,]
Auto.test=Auto[-train,]
mpgClass.train=mpgClass[train]
mpgClass.test=mpgClass[-train]

ResultBoosting.test <- matrix(1,48)
colnames(ResultBoosting.test) = "Test_Error_Rate"

for(i in seq(from=300,to=5000,by=100)){
  
  tree.Auto=gbm(mpgClass~., Auto.train, distribution="bernoulli",n.trees=i)
  tree.pred.prob=predict(tree.Auto, Auto.test, n.trees=i, type="response")
  tree.pred=ifelse(tree.pred.prob>0.5, "High", "Low")
  table(mpgClass.test, tree.pred)
  ResultLoop <- mean(tree.pred!=mpgClass.test)
  ResultBoosting.test[(i-200)/100,] <- c(ResultLoop)
  
}

# Find the index of the test error error matrix with the lowest error
ResultBoosting.test[which.min(ResultBoosting.test),] # Best test error = 0.04615385

# Number of trees producing best test error
300*which.min(ResultBoosting.test) # 300 trees

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Seed for official run: seed(3)

# Data Mining T6:20
# Group Project
# Victor Dechapanichkul, Taylor Starks, Tony Willett, Marwa Mustafa Alkhaleel

# This R script explores Naive Bayes classifcation models in order to classify automobiles as having a low or high MPG rating
# based on various automobile specifications

# Access needed libraries for classification analysis
library(e1071)
library(class)

x = read.table(file.choose(),header=F) # Select auto-mpg data file

names(x) = c("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name") # Add column labels to data set

table(is.na(x)) # Check for missing values, 14 missing values

Auto = x[complete.cases(x),]  # Delete all missing values

attach(Auto) # Attach Auto data frame to R

Auto <- Auto[,-9]; # remove Car Name column

mpgClass = ifelse(mpg<=23,"Low","High") # Split continuous mpg (response) variable at 23 (closest integer value to both mean and median of mpg variable) to convert it to a class label

Auto=data.frame(Auto, mpgClass) # Merge binary mpgClass variable with original data set

Auto=Auto[,-1] # Remove original, continuous mpg variable

# Split full data set into training data set (2/3 = 262) and test data set
set.seed(3)
train=sample(1:nrow(Auto), 262)
Auto.train=Auto[train,]
Auto.test=Auto[-train,]
mpgClass.train=mpgClass[train]
mpgClass.test=mpgClass[-train]

# Use naiveBayes function to create a class prediction model based on the training set
Naive_Bayes_Model=naiveBayes(mpgClass~., Auto.train)

# Training Error Rate
NB_Predictions=predict(Naive_Bayes_Model,Auto.train) # Prediction on the training dataset
table(NB_Predictions,mpgClass.train) # Confusion matrix to check accuracy
mean(NB_Predictions!=mpgClass.train) # Training error rate 0.1030534

#Test Error Rate
NB_Predictions=predict(Naive_Bayes_Model,Auto.test) # Prediction on the test dataset
table(NB_Predictions,mpgClass.test) # Confusion matrix to check accuracy
mean(NB_Predictions!=mpgClass.test) # Test error rate 0.1153846

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Seed for official run: seed(3)

# Data Mining T6:20
# Group Project
# Victor Dechapanichkul, Taylor Starks, Tony Willett, Marwa Mustafa Alkhaleel

# This R script explores support vector machine classifcation models in order to classify automobiles as having a low or high MPG rating
# based on various automobile specifications

# Access needed libraries for classification analysis
library(class)
library(e1071)

x = read.table(file.choose(),header=F) # Select auto-mpg data file

names(x) = c("mpg","cylinders","displacement","horsepower","weight","acceleration","model year","origin","car name") # Add column labels to data set

table(is.na(x)) # Check for missing values, 14 missing values

Auto = x[complete.cases(x),]  # Delete all missing values

attach(Auto) # Attach Auto data frame to R

Auto <- Auto[,-9]; # remove Car Name column

mpgClass = ifelse(mpg<=23,"Low","High") # Split continuous mpg (response) variable at 23 (closest integer value to both mean and median of mpg variable) to convert it to a class label

Auto=data.frame(Auto, mpgClass) # Merge binary mpgClass variable with original data set

Auto=Auto[,-1] # Remove original, continuous mpg variable

# Split full data set into training data set (2/3 = 262) and test data set
set.seed(3)
train=sample(1:nrow(Auto), 262)
Auto.train=Auto[train,]
Auto.test=Auto[-train,]
mpgClass.train=mpgClass[train]
mpgClass.test=mpgClass[-train]

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Linear SVM--------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a linear support vector classifier from the training set
# Invoke the tune function to find the best cost
tune.Auto.L=tune(svm, mpgClass~., data=Auto.train, kernel="linear", ranges=list(cost=c(0.001*(1:9),0.01*(1:9),0.1*(1:9),1*(1:9),10*(1:9),100)))

# Store best model as an object (cost = 0.04)
bestmod.Auto.L=tune.Auto.L$best.model

# Training error rate for best model
BM.train.L=predict(bestmod.Auto.L, Auto.train) # Prediction on the training dataset
table(BM.train.L, mpgClass.train) # Confusion matrix to check accuracy
mean(BM.train.L!=mpgClass.train) # Training error rate = 0.08778626

# Test error rate for best model
BM.test.L=predict(bestmod.Auto.L, Auto.test) # Prediction on the test dataset
table(BM.test.L, mpgClass.test) # Confusion matrix to check accuracy
mean(BM.test.L!=mpgClass.test) # Test error rate = 0.08461538

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Radial SVM--------------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a radial support vector classifier from the training set
# Invoke the tune function to find the best cost
tune.Auto.R=tune(svm, mpgClass~., data=Auto.train, kernel="radial", ranges=list(cost=c(0.001*(1:9),0.01*(1:9),0.1*(1:9),1*(1:9),10*(1:9),100), gamma=c(0.001*(1:9),0.01*(1:9),0.1*(1:9),1*(1:9),10*(1:9),100)))

# Store best model as an object (cost = 0.8, gamma = 0.5)
bestmod.Auto.R=tune.Auto.R$best.model

# Training error rate for best model
BM.train.R=predict(bestmod.Auto.R, Auto.train) # Prediction on the training dataset
table(BM.train.R, mpgClass.train) # Confusion matrix to check accuracy
mean(BM.train.R!=mpgClass.train) # Training error rate = 0.03053435

# Test error rate for best model
BM.test.R=predict(bestmod.Auto.R, Auto.test) # Prediction on the test dataset
table(BM.test.R, mpgClass.test) # Confusion matrix to check accuracy
mean(BM.test.R!=mpgClass.test) # Test error rate = 0.09230769

#------------------------------------------------------------------------------------------------------------------------------------------------------------------
#Polynomial SVM----------------------------------------------------------------------------------------------------------------------------------------------------
#------------------------------------------------------------------------------------------------------------------------------------------------------------------

# Create a polynomial support vector classifier from the training set
# Invoke the tune function to find the best cost
tune.Auto.P=tune(svm, mpgClass~., data=Auto.train, kernel="polynomial", ranges=list(cost=c(0.001*(1:9),0.01*(1:9),0.1*(1:9),1*(1:9),10*(1:9),100), gamma=c(0.001*(1:9),0.01*(1:9),0.1*(1:9),1*(1:9),10*(1:9),100)))

# Store best model as an object (cost = 100, gamma = 0.08)
bestmod.Auto.P=tune.Auto.P$best.model

# Training error rate for best model
BM.train.P=predict(bestmod.Auto.P, Auto.train) # Prediction on the training dataset
table(BM.train.P, mpgClass.train) # Confusion matrix to check accuracy
mean(BM.train.P!=mpgClass.train) # Training error rate = 0.04580153

# Test error rate for best model
BM.test.P=predict(bestmod.Auto.P, Auto.test) # Prediction on the test dataset
table(BM.test.P, mpgClass.test) # Confusion matrix to check accuracy
mean(BM.test.P!=mpgClass.test) # Test error rate = 0.06923077
