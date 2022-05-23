#import necessary packages
library("readxl")   #to read excel file
library(infotheo)   #to use 
library(praznik)
library(class)
library(e1071)
library(caTools)
library(caret)
library(MASS)


#set seed
set.seed(2022)

#function to normalize data, important for knn
normal = function(x){
  return((x-min(x))/(max(x)-min(x)))
}

#function to split the data randomly into training and test (70/30)
split_data = function(data){
  
  #initialize list to be returned
  out = list()
  
  data_norm = lapply(data[,1:16], normal)
  data[,1:16] = data.frame(data_norm)
  
  #draw randoms samples
  indices = 1:nrow(data)
  training_indices = round(dim(data)[1]*0.7)
  training_indices = sample(0:nrow(data), training_indices, replace=FALSE)
  
  #split set
  `%notin%` = Negate(`%in%`)
  data_training = data[row.names(data) %in% training_indices, ]
  data_test = data[row.names(data) %notin% training_indices, ]
  
  #prepare output
  out$training = data_training
  out$test = data_test
  
  return(out)
}

#function that returns selected features based on one of three features selection methods
feature_selection = function(data, method, n_features, return_scores = FALSE){
  #takes data and the method to calc the mutual information as input
  
  #initialize dataframe to store features and mutual information scores
  columns = c('features', method)
  mi_scores = data.frame(matrix(nrow = ncol(data)-1, ncol = length(columns)))
  colnames(mi_scores) = columns
  mi_scores$features = names(data)[1:16]
  features = names(data)[1:16]
  
  #discretizise data for mutual information calculations
  data_disc = discretize(data,"equalwidth", nbins=(nrow(data)^(0.5)))
  
  if (method == 'mi'){
    for (i in 1:16){
      mi_scores$mi[i] = mutinformation(data_disc[,i],data_disc[,17])
      mi_scores = mi_scores[order(mi_scores$mi, decreasing = TRUE),]
    }
  }
  
  if (method == 'jmi'){
    JMI_tmp = JMI(data_disc[,1:16],data_disc[,17], k=16)
    rownames(mi_scores) = JMI_tmp$selection
    for (i in 1:16){
      mi_scores$jmi[i] = JMI_tmp$score[i]
      mi_scores$features[i] = features[JMI_tmp$selection[i]]
    }
  }
  
  if (method == 'cmim'){
    CMIM_tmp = CMIM(data_disc[,1:16], data_disc[,17], k=16)
    rownames(mi_scores) = CMIM_tmp$selection
    for (i in 1:16) {
      mi_scores$cmim[i] = CMIM_tmp$score[i]
      mi_scores$features[i] = features[CMIM_tmp$selection[i]]
    }
  }
  
  #select only the first n_features
  mi_scores = mi_scores[1:n_features,]
  
  #check if one wants also the mi scores returned
  if (!return_scores){
    mi_scores = mi_scores$features
  }
  
  return(mi_scores)
}

#knn classifier as function, calc_perf function used
knn_classifier = function(training_data, test_data, class){
  #run knn algorithm
  knn_classified = knn(training_data[1:ncol(training_data)-1],test_data[1:ncol(test_data)-1],class,k=3)
  
  #calc and return performance metrics
  knn_perf = calc_perf(knn_classified, test_data$Class)
  return(knn_perf)
}

#naive bayes classifier as function, calc_perf function used
nb_classifier = function(training_data, test_data, class){
  #run naive bayes algorithm
  classifier_cl = naiveBayes(class ~ ., data = training_data[1:ncol(training_data)-1])
  nb_classified = predict(classifier_cl, newdata = test_data[1:ncol(test_data)-1])
  
  #calc and return performance metrics
  nb_perf = calc_perf(nb_classified, test_data$Class)
  return(nb_perf)
}

#perform pca, select number of features such that explained variance is over threshold
pc_analysis = function(reduce_training_data, reduce_test_data, thres){
  #inizialize return list
  pca_out = list()
  #perform pca
  db_pca_training <- prcomp(reduce_training_data[1:ncol(reduce_training_data)-1], scale = TRUE)
  
  #calculate explained variance
  var_explained = db_pca_training$sdev^2 / sum(db_pca_training$sdev^2)
  
  i=1
  while(sum(var_explained[1:i])<thres){i = i+1}
  
  cat("\nwe selected",i,"principal components which explain",sum(var_explained[1:i]),'percent of the total variance')
  db_pca_training = (-1)*db_pca_training$x[,1:i]
  pca_out$training = data.frame(db_pca_training)
  db_pca_test = prcomp(reduce_test_data[1:ncol(reduce_test_data)-1], scale=TRUE)
  db_pca_test = (-1)*db_pca_test$x[,1:i]
  pca_out$test = data.frame(db_pca_test)
  return(pca_out)
}

#perform lda
ld_analysis = function(training_data, test_data, data){
  lda_out = list()
  
  lda_model_training <- lda(training_data$Class~., data=training_data[1:ncol(training_data)-1])
  lda_training = predict(lda_model_training)$x
  lda_out$training = data.frame(lda_training)
  
  lda_model_test = lda(test_data$Class~., data = test_data)
  lda_test = predict(lda_model_test)$x
  lda_out$test = data.frame(lda_test)
  return(lda_out)
}

#calc per function which calculates the in 5.) defined performance metrics
calc_perf = function(db_pred, test_data_class){
  #returns acc, recall, precision and f1 in this order and takes predicted classes of the classifier as input
  
  #calculate TP, FP, TN, FN for all bean classes
  db_classes = data.frame(unique(test_data_class))
  db_perf = data.frame(db_pred, test_data_class)
  db_TP = numeric(0)
  db_FP = numeric(0)
  db_TN = numeric(0)
  db_FN = numeric(0)
  i=1
  
  for (db_class in unique(test_data_class)){
    #TP (seker predicted and was seker)
    db_TP[i] = sum(db_perf[db_perf$test_data_class == db_class,]$db_pred == db_class)
    #FP (seker predicted but was not seker)
    db_FP[i] = sum(db_perf[db_perf$test_data_class != db_class,]$db_pred == db_class)
    #TN (not seker predicted and was not seeker)
    db_TN[i] = sum(db_perf[db_perf$test_data_class != db_class,]$db_pred != db_class)
    #FN (not seker predicted but was seker)
    db_FN[i] = sum(db_perf[db_perf$test_data_class == db_class,]$db_pred != db_class)
    i = i+1
  }
  
  db_classes$TP = db_TP
  db_classes$FP = db_FP
  db_classes$TN = db_TN
  db_classes$FN = db_FN
  
  #calculate precision, recall and f1 scores for each class
  db_recall = numeric(0)
  db_precision = numeric(0)
  db_f1 = numeric(0)
  i=1
  for (db_class in unique(test_data_class)){
    #recall (TP/TP+FN)
    db_recall[i] = db_classes[i,2]/(db_classes[i,2]+db_classes[i,5])
    #precision (TP/TP+FP)
    db_precision[i] = db_classes[i,2]/(db_classes[i,2]+db_classes[i,3])
    #f1 (2*(precision*recall)/(precision+recall))
    db_f1[i] = 2*(db_recall[i]*db_precision[i])/(db_recall[i]+db_precision[i])
    i = i+1
  }
  
  db_classes$recall = db_recall
  db_classes$precision = db_precision
  db_classes$f1 = db_f1
  
  #return (arithmetic) mean of performance measure
  return(c(sum(db_pred == test_data_class)/length(test_data_class),mean(db_classes$recall),mean(db_classes$precision),mean(db_classes$f1)))
}

main = function(){
  #import data
  dry_bean = read_excel("Dry_Bean_Dataset.xlsx")
  
  #split data
  data = split_data(dry_bean)
  training_data = data$training
  test_data = data$test
  
  #select features based on mutual information scores
  #change the second input (mi, cmim, jmi) and the last input (3,5,10,15) to reproduce results from the report
  features_mi = feature_selection(training_data, 'cmim', 3)
  training_data_mi = training_data[,c(features_mi,'Class')]
  test_data_mi = test_data[,c(features_mi,'Class')]
  
  #run knn classifier with selected features and returns accuracy, macro recall, macro precision and macro f1 scores
  result_mi_knn = knn_classifier(training_data_mi, test_data_mi, training_data_mi$Class)
  cat('\nmi knn, acc:', result_mi_knn[1], 'and recall:',result_mi_knn[2][1], 'and precision:',result_mi_knn[3][1], 'and f1:',result_mi_knn[4][1])
  
  #run naive bayes classifier with selected features and returns accuracy, macro recall, macro precision and macro f1 scores
  result_mi_nb = nb_classifier(training_data_mi, test_data_mi, training_data_mi$Class)
  cat('\nmi naive bayes, acc:', result_mi_nb[1], 'and recall:',result_mi_nb[2][1], 'and precision:',result_mi_nb[3][1], 'and f1:',result_mi_nb[4][1])
  
  #pca which returns reduced dataframe
  data_pca = pc_analysis(training_data,test_data,0.8)
  training_data_pca = data_pca$training
  training_data_pca$Class = training_data$Class #need another column since knn and nb classifier cut last column
  test_data_pca = data_pca$test
  test_data_pca$Class = test_data$Class #need another column since knn and nb classifier cut last column
  
  #run knn classifier with selected features and returns accuracy, macro recall, macro precision and macro f1 scores
  result_pca_knn = knn_classifier(training_data_pca, test_data_pca, training_data$Class)
  #run naive bayes classifier with selected features and returns accuracy, macro recall, macro precision and macro f1 scores
  result_pca_nb = nb_classifier(training_data_pca, test_data_pca, training_data$Class)
  cat('\npca knn, acc:', result_pca_knn[1], 'and recall:',result_pca_knn[2][1], 'and precision:',result_pca_knn[3][1], 'and f1:',result_pca_knn[4][1])
  cat('\npca naive bayes, acc:', result_pca_nb[1], 'and recall:',result_pca_nb[2][1], 'and precision:',result_pca_nb[3][1], 'and f1:',result_pca_nb[4][1])
  
  #lda which returns reduced dataframe
  data_lda = ld_analysis(training_data, test_data)
  training_data_lda = data_lda$training
  training_data_lda$Class = training_data$Class
  test_data_lda = data_lda$test
  test_data_lda$Class = test_data$Class
  
  result_lda_knn = knn_classifier(training_data_lda, test_data_lda, training_data$Class)
  result_lda_nb = nb_classifier(training_data_lda, test_data_lda, training_data$Class)
  cat('\nlda knn, acc:', result_lda_knn[1], 'and recall:',result_lda_knn[2][1], 'and precision:',result_lda_knn[3][1], 'and f1:',result_lda_knn[4][1])
  cat('\nlda naive bayes, acc:', result_lda_nb[1], 'and recall:',result_lda_nb[2][1], 'and precision:',result_lda_nb[3][1], 'and f1:',result_lda_nb[4][1])
  
}

main()
