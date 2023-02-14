# Classification Modeling 

### APPROACH 1: 
# these functions output raw probabilities there's another function that takes in raw probabilites + a threshold which returns predictions 
# these functions also take as input individual training/testing data sets as opposed to the entire list returned by split


#######################################################################################################################################

## This R Script contains functions to run models as well as well as helper fns to evaluate them 
## These functions apply only to binary classification (we'll send over separate fns for multiclass)

## helper functions: 
# split
# finding optimal threshold
# confusion matrix
# knn helper to find kbest
# getting predictions
# evaluating 

## models that are running
# logisitic regression
# knn 
# naive bayes
# random forest
# decision trees
# xgboost 



## splitting function

split = function(df, proportion, respCol){
  set.seed(22)
  sample = sample(1:nrow(df), split * nrow(df))
  
  pred.train = df[sample, -respCol]
  pred.test = df[-sample, -respCol]
  resp.train = df[sample, respCol]
  resp.test = df[-sample, respCol]
  
  return(list(pred.train, pred.test, resp.train, resp.test))
}

## finding optimal threshold
optimalThreshold = function(roc){
  youden = roc$specificities + roc$sensitivities - 1
  index = which(youden == max(youden))
  threshold = roc$thresholds[index]
  
  return(threshold)
  
}


## confusion matrix 

cm = function(pred, resp.test){
  (table(pred,resp.test))
}

## run logistic regression

runLogit = function(pred.train, pred.test, resp.train){
  #training
  print("training logistic regression...")
  glm1.out = glm(resp.train~., pred.train, family = binomial)
  
  #testing
  # return prob
  glm1.prob = predict(glm1.out, newdata = pred.test, type = "response")
  return(glm1.prob)
}

## KNN
### kBest Calculation 
kBest = function(pred.train, pred.test, resp.train, min = 20, max = 50){
  suppressWarnings(suppressMessages(library(FNN)))
  k.max = max
  mrc.k = rep(NA, max - min)
  for ( kk in min:k.max ){
    # change brute to kdtree if sample size > 10,000
    knn.out = knn.cv(train=pred.train,cl =resp.train,k=kk,algorithm="kdtree")
    mcr.k[kk - (min - 1)] = sum(knn.out != resp.train)/(length(resp.train))
  }
  
  return(which(mcr.k == min(mcr.k))[0])
}
### run knn

runKNN = function(pred.train, pred.test, resp.train){
  suppressWarnings(suppressMessages(library(FNN)))
  
  #finding optimal k
  min = 20
  max = 50
  print("finding optimal k for knn...")
  k.best = kBest(pred.train, pred.test, resp.train)
  
  # making sure k isn't "on a bound" 
  while(abs(k.best - min) < 3 | abs(k.best - max) < 3){
    if(abs(k.best - min) < 3){
      min =  0
      max = k.best + 5
      k.best = kBest(pred.train, pred.test, resp.train)
    }
    
    if(abs(k.best - max) < 3){
      max = max + 5
      min = k.best - 5
      k.best = kBest(pred.train, pred.test, resp.train)
    }
  }
  
  
  #change brute to kdtree if sample size > 10,000
  print("training knn...")
  knn.pred = knn(train=pred.train,test=pred.test,cl=resp.train,k=k.best,algorithm="kdtree", prob = TRUE)
  
  # return prob
  knn.prob = attributes(knn.pred)$prob
  #knn.roc = roc(resp.test, knn.prob)
  return(knn.prob)
}

## run naive bayes

runNB = functio(pred.train, pred.test, resp.train){
  suppressWarnings(suppressMessages(library(e1071)))
  print("training naive bayes...")
  nb.out = naiveBayes(resp.train~.,data=pred.train)
  
  # return prob
  nb.pred = predict(nb.out,newdata=pred.test,type="raw") 
  nb.prob = predict(nb.out,newdata=pred.test,type="raw")[,2]
  #nb.roc = roc(resp.test, nb.prob)
  return(nb.prob)
}

## run random forest

#scale # of trees down with sample size 
runRF = function(pred.train, pred.test, resp.train){
  suppressMessages(library(randomForest))
  rf.out = randomForest::randomForest(resp.train~., data = pred.train, importance = TRUE)
  #return prob
  rf.prob = predict(rf.out, newdata = pred.test, type = "prob")[,2]
  #rf.roc = roc(resp.test, rf.pred)
  return(rf.prob)
}


## run decision tree

runDT = function(pred.train, pred.test, resp.train){
  suppressWarnings(suppressMessages(library(rpart)))
  suppressWarnings(suppressMessages(library(rpart.plot)))
  rpart.out = rpart(resp.train~.,data=pred.train)
  
  #return prob
  rpart.prob = predict(rpart.out,newdata=pred.test,type="prob")[,2]  # probability of class 1
  rpart.plot(rpart.out)
  # rpart.roc = roc(resp.test, rpart.prob)
  return(rpart.prob)
}

## run xgboost 

runXgboost = function(x, response){
  suppressMessages(library(xgboost))
  train.matrix.new = xgb.DMatrix(data = as.matrix(pred.train), 
                                 label = as.numeric(as.factor(resp.train))-1)
  test.matrix.new = xgb.DMatrix(data = as.matrix(pred.test), 
                                label = as.numeric(as.factor(resp.test))-1)
  
  new.cv.out = suppressWarnings(xgb.cv(data = train.matrix.new, 
                                       params=list(objective="binary:logistic"), 
                                       nfold = 5, nrounds = 30, verbose = 0))
  
  xg.out = xgboost(data = train.matrix.new,
                   nrounds = which.min(new.cv.out$evaluation_log$test_logloss_mean),
                   params=list(objective="binary:logistic"),
                   verbose = 0)
  
  #return prob
  xg.prob = predict(xg.out, newdata = test.matrix.new, type = "response")
  #xg.roc = roc(resp.test, xg.prob)
  return(xg.prob)
  
  ## getting predictions 
  
  predict = function(df, response, prob, threshold){
    resp = df[response]
    class1 = unique(resp)[1]
    class2 = unique(resp)[2]
    
    pred = rep(class1, length(df))
    pred[prob > threshold] = class2
    
    return(pred)
    
  }
  
  ## evaluation (returning mcr,auc, 
  evaluate = function(pred, roc, resp.test){
    #pred[prob > threshold] = class2
    #cm = table(pred, resp.test)
    mcr = sum(pred != resp.test)/(length(resp.test))
    auc = roc$auc
    return(c(mcr,auc)) 
    
  })