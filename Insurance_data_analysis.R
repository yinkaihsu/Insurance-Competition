# Install packages we need
#install.packages("readr")   #Read Rectangular Text Data
#install.packages("dplyr")   #A Grammar of Data Manipulation
#install.packages("rpart")
#install.packages("rpart.plot")
#install.packages("partykit")
#install.packages("arules")
#install.packages("arulesViz")

# Call out the installed packages
library(readr)
library(dplyr)
library(rpart)
library(rpart.plot)
library(partykit)
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
inspect(sort(rulesOwn, by = "lift"))
#sink()

# Save the algorithm with data fitted
# save.image(file = "Insurance_ARules.RData")
