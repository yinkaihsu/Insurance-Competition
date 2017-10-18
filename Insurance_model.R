# Install packages we need
#install.packages("readr")   #Read Rectangular Text Data
#install.packages("dplyr")   #A Grammar of Data Manipulation
#install.packages("glmnet")  #Elastic-Net Regularized Generalized Linear Models
#install.packages("Matrix")
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("partykit")
#install.packages("caret")
#install.packages("e1071")
#install.packages("nnet")
#install.packages("arules")
#install.packages("arulesViz")

# Call out the installed packages
library(readr)
library(dplyr)
library(glmnet)
library(rpart)
library(rpart.plot)
library(partykit)
library(caret)
library(e1071)
library(nnet)
library(arules)
library(arulesViz)

# Read data
insurance <- read_csv("C:/Users/ian19_000/Downloads/01.2011_Census_Microdata.csv")
#str(insurance)

# Original name of data
#[1] "Person ID"                  "Region"                    "Residence Type"            "Family Composition"
#[5] "Population Base"            "Sex"                       "Age"                       "Marital Status"
#[9] "Student"                    "Country of Birth"          "Health"                    "Ethnic Group"
#[13] "Religion"                  "Economic Activity"         "Occupation"                "Industry"
#[17] "Hours worked per week"     "Approximated Social Grade"

# Rename the variables
names = c("PersonID", "Region", "ResidenceType", "FamilyComposition", "PopulationBase", "Sex",
          "Age", "MaritalStatus", "Student", "CountryOfBirth","Health", "EthnicGroup",
          "Religion", "EconomicActivity", "Occupation", "Industry", "HoursWorked", "SocialGrade")
colnames(insurance) = names

# Discard Health = -9
insurance <- insurance %>% filter(Health != -9)
#str(insurance)

# Change variables into factor
insurance <- insurance %>% select(-PersonID, -ResidenceType, - PopulationBase) %>% 
  mutate_all(funs(as.factor(.)))
#str(insurance)


### Decision Tree with loss matrix =======================================================
# Define loss matrix
lossmatrix_first = matrix(c(  0,  1,  1,  1,  1,
                              2,  0,  2,  2,  2,
                              6,  6,  0,  6,  6,
                              24, 24, 24,  0, 24,
                              120,120,120,120,  0), byrow = T, nrow = 5)

lossmatrix_second = matrix(c( 0, 1, 2, 6,24,
                              1, 0, 1, 2, 6,
                              2, 1, 0, 1, 2,
                              6, 2, 1, 0, 1,
                              24, 6, 2, 1, 0), byrow = T, nrow = 5)

lossmatrix = lossmatrix_first * t(lossmatrix_second)

# Spilt data into training set and testing set
set.seed(1234)
idx_fold <- sample(1:5, nrow(insurance), replace = T)
idc_train <- idx_fold != 5


# Use loss matrix to construct Decision Tree model
rpart_fit <- rpart(Health ~ ., data = insurance, subset = idc_train, method = "class",
                   parms = list(loss = lossmatrix), cp = -1)

# Print the result of Decision Tree model
#sink("Insurance_Tree_model_full.log")
#rpart_fit
#sink()

# Plot model (method 1)
#prp(rpart_fit, faclen = 0, fallen.leaves = T, extra = 2)

# Plot model (method 2)
#party_fit <- as.party(rpart_fit)
#plot(party_fit)

# Find the predicted class
pred_train <- predict(rpart_fit, insurance[idc_train, ], type = "class")

pred_test <- predict(rpart_fit, insurance[!idc_train, ], type = "class")

# See the prediction
table_train = table(real = insurance[idc_train, ]$Health, predict = pred_train)
table_train
#    predict
#real     1     2     3     4     5
#1    74480 86745 40252  8218  2294
#2    15385 79885 34922 17068  5994
#3      973 10622 26458 14624  6995
#4       19   292  3258 12751  3438
#5        0    12   131  1045  4538

table_test = table(real = insurance[!idc_train, ]$Health, predict = pred_test)
table_test
#    predict
#real     1     2     3     4     5
#   1 15606 23662 10529  2485   700
#   2  6342 16355  9663  4539  1591
#   3   868  3535  4399  4062  1944
#   4   119   594  1160  1838  1089
#   5    41   125   294   592   406

# Calculate the classification error
error_train = 1 - sum(diag(table_train)) / sum(table_train)
error_train
#[1] 0.5601411

error_test = 1 - sum(diag(table_test)) / sum(table_test)
error_test
#[1] 0.6569692

# Find the best CP under 10-fold CV
cp_matrix = printcp(rpart_fit)
cp_matrix = cp_matrix[c(-1,-2),]
cp_best = cp_matrix[which.min(cp_matrix[,"xerror"]), "CP"]

# Use the best CP to construct model
prune_fit <- prune(rpart_fit, cp = cp_best)

# Print the result of model after pruning
#sink("Insurance_Tree_model_prune.log")
#prune_fit
#sink()

# Call variable importance
prune_fit$variable.importance
# EconomicActivity               Age     MaritalStatus        Occupation          Industry       SocialGrade 
#       55125.1060        37957.9889        11544.6842         3325.5436         3245.3986         2575.8693 
#FamilyComposition          Religion            Region       HoursWorked       EthnicGroup           Student 
#        1853.0231         1777.5423         1757.8407         1338.4296          650.0216          255.0316 
#              Sex    CountryOfBirth 
#         235.3054          197.4269

# Find the predicted class
pred_prune_test <- predict(prune_fit, insurance[!idc_train, ], type = "class")

# See the prediction
table_prune_test = table(real = insurance[!idc_train, ]$Health, predict = pred_prune_test)
table_prune_test
#    predict
#real     1     2     3     4     5
#   1   415 30853 19306  2307   101
#   2    84 14535 17011  6442   418
#   3    10  1949  5211  6529  1109
#   4     2   251   936  2529  1082
#   5     0    59   198   786   415

# Calculate the classification error
error_prune_test = 1 - sum(diag(table_prune_test)) / sum(table_prune_test)
error_prune_test
#[1] 0.7946916

# Print the evaulation result
#sink("Insurance_Tree_result.log")
#lossmatrix
#table_test;table_prune_test
#error_test;error_prune_test
#prune_fit$variable.importance
#sink()

# Save the algorithm with data fitted
#save.image(file = "Insurance_Tree.RData")


### Association Rules ================================================================
# Change into transaction format
insurance_trans <- as(insurance, "transactions")

# See the summary of data
summary(insurance_trans)
sort(itemFrequency(insurance_trans), decreasing = T)

# Find the rules with support >= 100/562937 and confidence >= 0.05
rules <- apriori(insurance_trans, parameter = list(maxlen = 5,
                                                   support = 100/562937, confidence = 0.05))
summary(rules)

#plot(rules, measure = c("confidence", "lift"), shading = "support")
#plot(rules, method = "grouped", control = list(gp_labels = gpar(cex = 0.3)))

# Choose the rules with rhs contains "Health=5" and lift > 1
rulesOwn <- subset(rules, subset = rhs %pin% "Health=5" & lift > 1)
summary(rulesOwn)

# Call out the rules
#sink("Insurance_ARules_rules.log")
#inspect(sort(rulesOwn, by = "lift"))
#sink()

# Save the algorithm with data fitted
#save.image(file = "Insurance_ARules.RData")


## Multinomial Logistic Regression ==================================================
multinom_fit <- multinom(Health ~ ., data = insurance)

# Save the algorithm with data fitted
#save.image(file = "Insurance_MLR.RData")

# Save the coefficients of MLR
#sink("Insurance_MLR_coef.log")
#multinom_fit
#sink()

# # Calculate the p-value
# # IT TAKES TOO MUCH TIME!!!
# coeftest(multinom_fit)

# # Another method to fit
# # IT TAKES TOO MUCH TIME!!!
# y <- insurance$Health
# x <- Matrix::sparse.model.matrix(Health ~ (.) - 1, data = insurance)
# glm_fit <- glmnet(x, y, lambda = 0, family = "multinomial", penalty.factor = c(rep(0, 16)))


### Decision Tree (cp = 0.01) ==========================================================
# Use cp = 0.01 to construct CART model
rpart_fit_1 <- rpart(Health ~ ., data = insurance, method = "class",
                     parms = list(split = "information"), cp = 0.01)

# Plot CART model (method 1)
prp(rpart_fit_1, faclen = 0, fallen.leaves = T, extra = 2)
# Plot CART model (method 2)
party_fit_1 <- as.party(rpart_fit_1)
plot(party_fit_1)

# Find the predicted class
pred_1 <- predict(rpart_fit_1, insurance, type = "class")

# See the prediction
table_1 = table(real = insurance$Health, predict = pred_1)
table_1
#    predict
#real      1      2      3      4      5
#   1 207526  53811   2953    681      0
#   2  94394  82635  12475   2240      0
#   3  13841  37253  17638   5748      0
#   4   2146   9315   6248   6849      0
#   5    544   2222   1945   2473      0

# Calculate the classification error
error_1 = 1 - sum(diag(table_1)) / sum(table_1)
error_1
#[1] 0.44106

# Save the algorithm with data fitted
#save.image(file = "Insurance_Tree_01.RData")


### Decision Tree (cp = 0.001) and Pruning (cp = 0.01523104) =======================
# Use cp = 0.001 to construct CART model
rpart_fit_2 <- rpart(Health ~ ., data = insurance, method = "class",
                     parms = list(split = "information"), cp = 0.001)

# Call variable importance
rpart_fit_2$variable.importance
#EconomicActivity               Age     MaritalStatus        Occupation          Industry       HoursWorked 
#      86019.5182        67137.6903        26540.3417         9963.9093         8454.1797         8174.1730 
#     SocialGrade FamilyComposition     ResidenceType 
#       1724.4593          363.4121          126.6995

# Plot CART model (method 1)
prp(rpart_fit_2, faclen = 0, fallen.leaves = T, extra = 2)
# Plot CART model (method 2)
party_fit_2 <- as.party(rpart_fit_2)
plot(party_fit_2)

# Find the predicted class
pred_2 <- predict(rpart_fit_2, insurance, type = "class")

# See the prediction
table_2 = table(real = insurance$Health, predict = pred_2)
table_2
#    predict
#real      1      2      3      4      5
#   1 207526  53811   2953    681      0
#   2  94394  82635  12475   2240      0
#   3  13841  37253  17638   5748      0
#   4   2146   9315   6248   6849      0
#   5    544   2222   1945   2473      0

# Calculate the classification error
error_2 = 1 - sum(diag(table_2)) / sum(table_2)
error_2
#[1] 0.44106

# Find the best cp under 5-fold CV
# CAUTIOUS: waste some time and memory
train_control <- trainControl(method ="cv", number = 5)
train_control.model <- train(Health ~ ., data = insurance,
                             method = "rpart", trControl = train_control, na.action = na.omit)

# Save the optimal value of cp
#sink("Insurance_Tree_cp.log")
#train_control.model
#cp = 0.01523104
#sink()

# Use cp = 0.01523104 to construct CART model
prune_fit <- prune(rpart_fit_2, cp = 0.01523104)

# Call variable importance
prune_fit$variable.importance
#EconomicActivity               Age     MaritalStatus        Occupation          Industry       HoursWorked 
#      86019.5182        67137.6903        26540.3417         9963.9093         8454.1797         8174.1730 
#     SocialGrade FamilyComposition     ResidenceType 
#       1724.4593          363.4121          126.6995

# Plot CART model (method 1)
prp(prune_fit, faclen = 0, fallen.leaves = T, extra = 2)
# Plot CART model (method 2)
party_prune_fit <- as.party(prune_fit)
plot(party_prune_fit)

# Find the predicted class
pred_prune <- predict(prune_fit, insurance, type = "class")

# See the prediction
table_prune = table(real = insurance$Health, predict = pred_prune)
table_prune
#    predict
#real      1      2      3      4      5
#   1 207526  53811   2953    681      0
#   2  94394  82635  12475   2240      0
#   3  13841  37253  17638   5748      0
#   4   2146   9315   6248   6849      0
#   5    544   2222   1945   2473      0

# Calculate the classification error
error_prune = 1 - sum(diag(table_prune)) / sum(table_prune)
error_prune
#[1] 0.44106

# Save the algorithm with data fitted
#save.image(file = "Insurance_Tree_02.RData")


### Decision Tree (cp = 0.0001) =========================================================
# Use cp = 0.0001 to construct CART model
rpart_fit_3 <- rpart(Health ~ ., data = insurance, method = "class",
                     parms = list(split = "information"), cp = 0.0001)

# Call variable importance
rpart_fit_3$variable.importance
#EconomicActivity               Age     MaritalStatus        Occupation          Industry       HoursWorked 
#    92519.518933      69298.898637      27182.787788      15385.106276      12456.146984      11481.005920 
#     SocialGrade           Student FamilyComposition     ResidenceType          Religion            Region 
#     7943.150849       1871.875645        787.215322        297.452328        192.236330        164.390861 
#     EthnicGroup               Sex    CountryOfBirth    PopulationBase 
#       73.721062         30.586666         21.908393          1.341324

# Plot CART model (method 1)
prp(rpart_fit_3, faclen = 0, fallen.leaves = T, extra = 2)
# Plot CART model (method 2)
party_fit_3 <- as.party(rpart_fit_3)
plot(party_fit_3)

# Find the predicted class
pred_3 <- predict(rpart_fit_3, insurance, type = "class")

# See the prediction
table_3 = table(real = insurance$Health, predict = pred_3)
table_3
#    predict
#real      1      2      3      4      5
#   1 210166  50336   4067    402      0
#   2  94427  80209  15810   1298      0
#   3  13236  34172  22656   4416      0
#   4   2045   7514   9075   5924      0
#   5    555   1645   2843   2141      0

# Calculate the classification error
error_3 = 1 - sum(diag(table_3)) / sum(table_3)
error_3
#[1] 0.4334091

# Save the algorithm with data fitted
#save.image(file = "Insurance_Tree_03.RData")


### Decision Tree (cp = 0.00001) =========================================================
# Use cp = 0.00001 to construct CART model
rpart_fit_4 <- rpart(Health ~ ., data = insurance, method = "class",
                     parms = list(split = "information"), cp = 0.00001)

# Call variable importance
rpart_fit_4$variable.importance
#EconomicActivity               Age     MaritalStatus        Occupation          Industry       HoursWorked 
#     93820.72025       70704.25752       29161.01416       19619.75694       17633.11144       12914.80460 
#     SocialGrade            Region FamilyComposition          Religion           Student       EthnicGroup 
#     10323.15362        4190.86789        3786.09917        2392.77566        2220.13594        1382.57078 
#             Sex     ResidenceType    CountryOfBirth    PopulationBase 
#       843.05299         700.22848         624.36331          31.49653 

# Plot CART model (method 1)
prp(rpart_fit_4, faclen = 0, fallen.leaves = T, extra = 2)
# Plot CART model (method 2)
party_fit_4 <- as.party(rpart_fit_4)
plot(party_fit_4)

# Find the predicted class
pred_4 <- predict(rpart_fit_4, insurance, type = "class")

# See the prediction
table_4 = table(real = insurance$Health, predict = pred_4)
table_4
#    predict
#real      1      2      3      4      5
#   1 213170  47087   4285    418     11
#   2  86881  89119  14470   1234     40
#   3  13476  32451  24941   3537     75
#   4   2200   7466   8988   5803    101
#   5    595   1676   2969   1766    178

# Calculate the classification error
error_4 = 1 - sum(diag(table_4)) / sum(table_4)
error_4
#[1] 0.4080847

# Save the algorithm with data fitted
#save.image(file = "Insurance_Tree_04.RData")
