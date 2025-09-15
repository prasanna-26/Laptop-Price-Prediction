#install.packages("plyr")
library(plyr)
#install.packages("tidyverse")
library(tidyverse)
#install.packages("ggplot2")
library(ggplot2)
#install.packages("dplyr")
library(dplyr)
#install.packages("tidyr")
library(tidyr)
#install.packages("caret")
library(caret)
#install.packages("MLmetrics")
library(MLmetrics)
#install.packages("glmnet")
library(glmnet)
#install.packages("caTools")
library(caTools)
#install.packages("corrplot")
library(corrplot)
library(scales)

#Import the data set
data<-read.table(file.choose(), sep="," , header = TRUE)

#-------------------------Exploratory Data Analysis--------------------
#----------------------------Data Cleaning-----------------------------

#Checking for missing data
sapply(data, function(x) sum(is.na(x)))

#Splitting 'Gpu' into 'GPU_Vendor' and 'GPU_Type'
data[c('GPU_Vendor','GPU_Type')] <- str_split_fixed(data$Gpu,' ', 2)

#Splitting 'Memory' into 'Memory' and 'Storage_Type'
data[c('Memory','Storage_Type')] <- str_split_fixed(data$Memory,' ', 2)
#Removing GB from 'Memory'
data$Memory <- gsub("GB", " ", data$Memory)
#Removing TB from 'Memory'
data$Memory <- gsub("TB", " ",data$Memory)

#Remove the 'GB' from 'RAM' column
data$Ram <- gsub("GB", " ", data$Ram)

#Remove the 'kg' from 'Weight' column
data$Weight <- gsub("kg", " ", data$Weight)

#Splitting original 'Cpu' column' into 4 necessary categories
data[c('a', 'b', 'c','d')]<- str_split_fixed(data$Cpu,' ',4)
#Merging three columns to 'CPU_Series'
data$CPU_Series <- paste(data$a, data$b, data$c)
#Separating 'cpu' column to retain speed 
data$Cpu_Speed <- gsub("GHz", "", sapply(strsplit(as.character(data$Cpu), " "), 
                                         function(x) tail(x, 1)))

#Splitting the 'ScreeResolution' column to retain 'dimensions'
data$Screen_dimensions <- substr(data$ScreenResolution, nchar(data$ScreenResolution)-9, 
                                 nchar(data$ScreenResolution))
#Removing the unnecessary characters from 'Screen_dimensions'
data$Screen_dimensions <- gsub("D","", data$Screen_dimensions)
data$Screen_dimensions <- gsub("l","", data$Screen_dimensions)
data$Screen_dimensions <- gsub("n","", data$Screen_dimensions)
#Splitting the dimensions into two columns 'screen_width' and 'screen_height'
data[c('screen_width', 'screen_height')]<- str_split_fixed(data$Screen_dimensions,'x',2)

#Removing the unnecessary columns
data<- data[-c(5,6,9,14,16:19,22)]

#converting required columns into numeric
data$Cpu_Speed<- as.numeric(data$Cpu_Speed)
data$Ram<- as.numeric(data$Ram)
data$Memory<-as.numeric(data$Memory)
data$Weight<- as.numeric(data$Weight)
data$screen_width <- as.numeric(data$screen_width)
data$screen_height <- as.numeric(data$screen_height)

#Summary of the cleaned data set
summary(data)

#creating a bar-plot for the number of companies
numcompanies<- as.data.frame(table(data$Company))
ggplot(data=numcompanies, aes(x=Var1, y=Freq)) +geom_bar(stat="identity", color = "black")+
  ggtitle("Number of Companies") + xlab("Company Names") + ylab("Number of companies")

#boxplot comparing Company and price
ggplot(data)+geom_boxplot(aes(x=Company,y=Price))+theme_classic()+ggtitle("Company VS Price") +
  xlab("Company Names") + ylab("Price (INR)")


#Histogram to observe the distribution of Target Variable - 'Price'
ggplot(data, aes(x=Price)) + geom_histogram(color="black", fill="paleturquoise2")+
  scale_x_continuous(labels = comma) + scale_y_continuous(labels = comma)+
  ggtitle("Distribution of the target variable - Price")+ xlab("Price (INR)") + ylab("Frequency")

#---------------------Split the data into train and test sets-----------------

#Random numbers generator
set.seed(678)

#Split data into train and test in the ratio of 80/20
splitdata <- createDataPartition(y = data$Price, p = .80, list = FALSE)

#Train data
train_dataset <- data[splitdata,]

#Test data
test_dataset <- data[-splitdata,]

#Convert the train data set to matrix
train_x <- model.matrix(Price ~ ., train_dataset)[,-1]

#Convert the test data set to matrix
test_x <- model.matrix(Price ~ ., test_dataset)[,-1]

# Remove the extra columns from the train matrix
extra_cols <- setdiff(colnames(train_x), colnames(test_x))
train_x <- train_x[, !(colnames(train_x) %in% extra_cols)]

# Remove the extra columns from the test matrix
extra_cols <- setdiff(colnames(test_x), colnames(train_x))
test_x<- test_x[, !(colnames(test_x) %in% extra_cols)]

#Assign the dependent variable to 'train_y' of train set
train_y <- train_dataset$Price

#Assign the dependent variable to 'test_y' of test set
test_y <- test_dataset$Price

#-----------------------------Multiple Linear Regression--------------------

#random number generator
set.seed(8989)

#correlation between the variables
cor(data[, unlist(lapply(data, is.numeric))])

#visualize correlation
C = cor(data[, unlist(lapply(data, is.numeric))])
corrplot(C)

#We need to find the most significant variables
model <- lm(formula = Price ~ ., data = train_dataset)
summary(model)

#Ram, Memory, Weight, Cpu_Speed are most significant with Price
model.new <- lm(Price ~ Cpu_Speed + Ram + Memory + Weight, data=train_dataset)
summary(model.new)

#Predicting the new regression model on the test data
pred2 <- predict(model.new,test_dataset)

#RMSE value - test data
rmse_val_test <- RMSE(pred2, test_dataset$Price)
rmse_val_test

#-----------------------------------Ridge regression----------------------------

#to seed the randomiser so that you can repeat the results
set.seed(345)

#perform k-fold cross-validation for lambda values
ridge <- cv.glmnet(train_x, train_y, nfolds = 10, alpha = 0)
ridge

#Plot the results
plot(ridge)

# Check the min and max lambda values
# MIN log lambda
log(ridge$lambda.min)
# MAX log lambda within 1 standard error from MIN
log(ridge$lambda.1se)

# choosing lambda min  
ridge_fit_min <- glmnet(train_x, train_y, alpha=0, lambda=ridge$lambda.min)

# Show ridge regression Model info for lambda = min
ridge_fit_min

# display ridge regression coefficients
coef(ridge_fit_min)

#Plot coefficients of ridge model
ridge_model <- glmnet(train_x, train_y, alpha = 0)
plot(ridge_model, xvar="lambda")

# Predict train data using generated model for ridge regression for lambda=min
TrainPrediction_min <- predict(ridge_fit_min, newx = train_x)

# Predict test data using generated model for ridge regression for lambda=min
TestPrediction_min <- predict(ridge_fit_min, newx = test_x)

#RMSE - train data for lambda = min
ridgetrain_rmse_min <- RMSE(TrainPrediction_min, train_y)
ridgetrain_rmse_min

#RMSE - test data for lambda = min
ridgetest_rmse_min <- RMSE(TestPrediction_min, test_y)
ridgetest_rmse_min

#R2 score - train set
R2_Score(TrainPrediction_min, train_y)

#R2 score - test data
R2_Score(TestPrediction_min, test_y)

#-----------------------------------LASSO Regression-------------------------

#Random numbers generator
set.seed(456)
#Cross validation for lambda values
lasso <- cv.glmnet(train_x, train_y, alpha = 1, nfolds = 10)
lasso

#Visualize cross validation results
plot(lasso)

#Minimum log value of lambda
log(lasso$lambda.min)

#Maximum log value of lambda with one standard error
log(lasso$lambda.1se)

#Run lasso regression for minimum lambda value
lassomodel_min <- glmnet(train_x, train_y, alpha = 1, lambda = lasso$lambda.min)
lassomodel_min

#Coefficients of lasso model with minimum lambda value
coef(lassomodel_min)

#Plot coefficients of lasso model
lasso_model <- glmnet(train_x, train_y, alpha = 1)
plot(lasso_model, xvar="lambda")

#Predicting train data set for lasso model with minimum lambda value
predL_trainmin <- predict(lassomodel_min, newx = train_x)

#Predicting test data set for lasso model with minimum lambda value
predL_testmin <- predict(lassomodel_min, newx = test_x)

#RMSE - train set
lasso_train_min <- RMSE(predL_trainmin, train_y)
lasso_train_min

#RMSE - test set
lasso_test_min <- RMSE(predL_testmin, test_y)
lasso_test_min

#R2 score - train set
R2_Score(predL_trainmin, train_y)

#R2 score - test set
R2_Score(predL_testmin, test_y)