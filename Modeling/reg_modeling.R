


### APPROACH 2:
# have functions input list outputted by split function and sep into train/test/pred/resp manually
# also output list of (resp.test, resp.pred) which can be put into MSE and ran 




## These functions apply only to regression

# TODO : write KNN function



split = function(df, proportion, respCol){
  set.seed(22)
  sample = sample(1:nrow(df), split * nrow(df))
  
  pred.train = df[sample, -respCol]
  pred.test = df[-sample, -respCol]
  resp.train = df[sample, respCol]
  resp.test = df[-sample, respCol]
  
  return(list(pred.train, pred.test, resp.train, resp.test))
}

#MSE : 
# Helper for MSE
mse = function(resp.list){
  resp.test = resp.list[[1]]
  resp.pred = resp.list[[2]]
  model.mse <- mean((resp.test - resp.pred)^2)
  return(model.mse)
}

#plot_residuals
# plots residuals given resp.test, resp.pred
# also returns test mse

# TAKES IN OUTPUT LIST FROM RETURN FUNCTIONS 
residuals = function(resp.list){
  resp.test = resp.list[[1]]
  resp.pred = resp.list[[2]]
  res = (resp.pred - resp.test)
  res.df = data.frame("fitted" = resp.pred, "residuals" = res)
  if (nrow(res.df) > 1000){
    sampled = sample(nrow(res.df), 1000)
    plot(res.df[sampled, ])
    abline()
  }
  else{
    plot(res.df)
    abline()
  }
  return(mse(resp.test, resp.pred))
}


## Linear Regression
# Params : dataset(list of 4)
# Return : list(resp.test, resp.pred)

#simple linear regression takes in output from train_test_split()
#include other forms of linear regression (another paramter for type of interaction)
# interaction

runLinReg = function(dataset){
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  lm.out = lm(resp.train ~ ., data = pred.train)
  resp.pred = predict(lm.out,newdata=pred.test)
  return(list(resp.test, resp.pred))
} 



## Best Subset Selection
#Takes in list_dataset, Information Criteria: {0,1} -> {AIC, BIC}
#Prints summary of bestModel and returns (resp.test, resp.pred)


runBestSubsetSelection = function(dataset, IC){
  input = ifelse(IC == 0, "AIC", "BIC")
  require(bestglm)
  require(ggplot2)
  suppressMessages(library(bestglm))
  suppressMessages(library(ggplot2))
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  df = data.frame(pred.train,"y"=resp.train)
  # names(df)[1 : (k-1)] = c("col1",.,"col last")
  
  # y = data.frame("y" = pred.train)
  #add potential check for columns
  # if (which(colnames(df.train) == "y") != NULL){
  # }
  colnames(df.train)
  cat("There should be predictors and the y (response)")
  bg.out = bestglm(df.train, family = gaussian, IC = input) 
  print(bg.out$BestModel)
  resp.pred = predict(bg.out$BestModel,newdata = pred.test)
  return(list(resp.test,resp.pred))
  # return(1)
}

## Lasso (not tested)
#Takes 2 arguments : dataset(list of 4), boolean(1/0)
#Returns (resp.pred)

runPenRegression = function(dataset, lasso){
  #lasso = 1 for lasso, 0 for ridge
  require(glmnet)
  suppressMessages(library(glmnet))
  
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  
  x = model.matrix(resp.train~.,pred.train)[,-1] ; y = resp.train
  # ifelse(lasso == 0, cat("Performing ridge regression"), cat("Performing lasso regression"))
  out.lasso = glmnet(x,y,alpha=1) 
  plot(out.lasso, xvar="lambda")
  set.seed(301)  # cv.glmnet() performs random sampling...so set the seed!
  
  cv = cv.glmnet(x,y, alpha = lasso)
  plot(cv)
  
  min = cv$lambda.min
  log_val = log(cv$lambda.min)
  
  cat("\n Optimal value of lambda =",min,"log(min) =",log_val, "These are the variables retained:" )
  coef(out.lasso,cv$lambda.min)
  
  x.test = model.matrix(resp.test~.,pred.test)[,-1]
  resp.pred = predict(out.lasso,s=cv$lambda.min,newx=x.test)
  return(list(resp.test,resp.pred))
}

## Decision Tree

runDTReg = function(dataset){
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  
  suppressWarnings(suppressMessages(library(rpart)))
  suppressWarnings(suppressMessages(library(rpart.plot)))
  rpart.out = rpart(resp.train~.,data=pred.train)
  resp.pred <- predict(rpart.out,newdata=pred.test)
  df = data.frame(resp.test,resp.pred)
  # r_mse<-mean((rpart.pred - resp.test)^2)
  rpart.plot(rpart.out,extra=101) 
  plotcp(rpart.out)
  # rpart.roc = roc(resp.test , rpart.prob)
  return(list(resp.test,resp.pred))
}

## Random Forest
runRFReg = function(dataset){
  #load packages
  suppressMessages(library(tidyverse))
  suppressMessages(library(randomForest))
  # read data
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  
  rf.out = randomForest(resp.train~.,data=pred.train,importance=TRUE)
  resp.pred = predict(rf.out,newdata=pred.test) 
  ggplot(data=data.frame("x"=resp.test,"y"=resp.pred),mapping=aes(x=x,y=y)) +
    geom_point(size=0.1,color="saddlebrown") + xlim(0,2) + ylim(0,2) +
    geom_abline(intercept=0,slope=1,color="red")
  
  
  varImpPlot(rf.out,type=1)  # type=1 means "only show %IncMSE plot
  rf.mse = round(mean((resp.pred-resp.test)^2),3)
  return(list(resp.test, resp.pred))
}



## Iterative Algorithms (?)
## XGBoost

runXgboostReg = function(dataset){
  suppressMessages(library(xgboost))
  # read data
  pred.train = dataset[[1]]
  pred.test = dataset[[2]]
  resp.train = dataset[[3]]
  resp.test = dataset[[4]]
  train = xgb.DMatrix(data=as.matrix(pred.train),label=resp.train)
  test = xgb.DMatrix(data=as.matrix(pred.test),label=resp.test)
  set.seed(101)
  xgb.cv.out = xgb.cv(params=list(objective="reg:squarederror"),train,nrounds=30,nfold=5,verbose=0) 
  cat("The optimal number of trees is ",which.min(xgb.cv.out$evaluation_log$test_rmse_mean))
  xgb.out = xgboost(train,nrounds=which.min(xgb.cv.out$evaluation_log$test_rmse_mean),params=list(objective="reg:squarederror"))
  resp.pred = predict(xgb.out,newdata=test)
  cat("MSE:", round(mean((resp.pred-resp.test)^2),3))
  return(list(resp.test,resp.pred))
}

