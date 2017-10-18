# Insurance-Competition

This is an insurance competition of Cathaylife, 2017.

2017 / 10 / 16

Created by <font color="#006699">**Ian Hsu**</font> in Insurance Competition of Cathaylife, 2017

---

## Data Preprocessing

### Step 0

Install some packages we need and read data.


``` r=
#install.packages("readr")      #Read Rectangular Text Data
#install.packages("dplyr")      #A Grammar of Data Manipulation

library(readr)
library(dplyr)

insurance <- read_csv("01.2011_Census_Microdata.csv")
#str(insurance)
```

### Step 1

Rename variables with no space containing.

``` r=
names = c("PersonID", "Region", "ResidenceType", "FamilyComposition", "PopulationBase", "Sex",
          "Age", "MaritalStatus", "Student", "CountryOfBirth","Health", "EthnicGroup",
          "Religion", "EconomicActivity", "Occupation", "Industry", "HoursWorked", "SocialGrade")
colnames(insurance) = names
```

### Step 2

Discard data with the "Health" condition is unknown.

``` r=
insurance <- insurance %>% filter(Health != -9)
#str(insurance)
```

### Step 3

Based on the plot with contribution of "ResidenceType" and "PopulationBase" in data, we found that nearly everyone has the same residence type and population base.
As a result, we discard these two variables and the unique identifier "PersonID" and change all variables into "factor" type.

``` r=
insurance <- insurance %>% select(-PersonID, -ResidenceType, -PopulationBase) %>% 
  mutate_all(funs(as.factor(.)))
#str(insurance)
```

---

## Data Analysis - Decision Tree

### Step 0
Install some packages we need to use with Decision Tree.

``` r=
#install.packages("rpart")      #An algorithm of Decision Tree
#install.packages("rpart.plot") #The visualization of "rpart" package
#install.packages("partykit")   #Another algorithm of Decision Tree

library(rpart)
library(rpart.plot)
library(partykit)
```

### Step 1
In the real world of insurance, it is more serious to predict sick people to healthy one than to predict healthy people to unhealthy one.
Therefore, we construct first loss matrix below.

``` r=
lossmatrix_first = matrix(c(  0,  1,  1,  1,  1,
                              2,  0,  2,  2,  2,
                              6,  6,  0,  6,  6,
                             24, 24, 24,  0, 24,
                            120,120,120,120,  0), byrow = T, nrow = 5)
```

Moreover, it is acceptable to misclassify the "Health" condition with only one level.
However, it seems unacceptable with the difference of levels increased.
Thus, we construct second loss matrix below.

``` r=
lossmatrix_second = matrix(c( 0, 1, 2, 6,24,
                              1, 0, 1, 2, 6,
                              2, 1, 0, 1, 2,
                              6, 2, 1, 0, 1,
                             24, 6, 2, 1, 0), byrow = T, nrow = 5)
```

Last but not least, we conduct ```lossmatrix_first``` with the transpose of ```lossmatrix_second```.

``` R=
lossmatrix = lossmatrix_first * t(lossmatrix_second)
```

### Step 2

We spilt data into training set and testing set with the proportion of 5:1.
Afterward, we construct Decision Tree model with the training set based on the loss matrix above and let the tree fully grows to the end.

``` r=
set.seed(1234)
idx_fold <- sample(1:5, nrow(insurance), replace = T)
idc_train <- idx_fold != 5

rpart_fit <- rpart(Health ~ ., data = insurance, subset = idc_train, method = "class",
                   parms = list(loss = lossmatrix), cp = -1)
```

### Step 3

Predict the classes on both training set and testing set with the model above.
Then, print the confusion matrix and classification error.

``` r=
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
```

### Step 4
Now, we try to prune the tree to avoid overfitting.
We used 10-fold cross validation to find out the best value of cp, and thus prune the fully grown tree.

``` r=
cp_matrix = printcp(rpart_fit)
cp_matrix = cp_matrix[c(-1,-2),]
cp_best = cp_matrix[which.min(cp_matrix[,"xerror"]), "CP"]

prune_fit <- prune(rpart_fit, cp = cp_best)
```
![](https://i.imgur.com/WyWcLOp.png)

### Step 5

Predict the classes on both testing set with the tree model after pruning.
Then, print the confusion matrix and classification error.

``` r=
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
```

---

## Data Evaluation - Decision Tree

We compare the results on the testing set with the fully grown tree and pruning tree. 

``` r=
table_test;table_prune_test
#    predict
#real     1     2     3     4     5
#   1 15606 23662 10529  2485   700
#   2  6342 16355  9663  4539  1591
#   3   868  3535  4399  4062  1944
#   4   119   594  1160  1838  1089
#   5    41   125   294   592   406
#    predict
#real     1     2     3     4     5
#   1   415 30853 19306  2307   101
#   2    84 14535 17011  6442   418
#   3    10  1949  5211  6529  1109
#   4     2   251   936  2529  1082
#   5     0    59   198   786   415

error_test;error_prune_test
#[1] 0.6569692
#[1] 0.7946916
```

We find out the feature importance of each variables in the pruning tree.

``` r=
prune_fit$variable.importance
# EconomicActivity               Age     MaritalStatus        Occupation          Industry       SocialGrade 
#       55125.1060        37957.9889        11544.6842         3325.5436         3245.3986         2575.8693 
#FamilyComposition          Religion            Region       HoursWorked       EthnicGroup           Student 
#        1853.0231         1777.5423         1757.8407         1338.4296          650.0216          255.0316 
#              Sex    CountryOfBirth 
#         235.3054          197.4269
```

### Result

1. Although the prediction error is higher with the pruning tree, our desired goal and result are closer to the pruning tree based on the confusion matrix we had.

2. The three variables "EconomicActivity", "Age", "MaritalStatus" seem to influence the most on "Health" condition.

3. "Sex" does not have a great impact on the condition of "Health", which is a primary factor of the insurance fee considered by the agents nowadays.

4. The prediction power is not very good on this data set, so we might conclude that it is important to have a look on additional medical report before evaluate the levels of risk using just social factors on the insured person.

---

## Data Analysis and Evaluation - Association Rules

### Step 0

Install some packages we need to use with Association Rules.

``` r=
#install.packages("arules")     #An algorithm of Association Rules
#install.packages("arulesViz")  #The visualization of "arules" package

library(arules)
library(arulesViz)
```

### Step 1

We need to change into transaction format.

``` r=
insurance_trans <- as(insurance, "transactions")
```

Next, we found the rules with the count >= 100 and confidence >= 0.05.

``` r=
rules <- apriori(insurance_trans,
                 parameter = list(maxlen = 5,
                                  support = 100/562937,
                                  confidence = 0.05))
#summary(rules)
```

### Step 2

Based on the special condition of insurance industry, insurancial agents mostly focus on the rules of unhealthy people.

As a result, we subset the rules with "Health=5" and lift > 1.

``` r=
rulesOwn <- subset(rules, subset = rhs %pin% "Health=5" & lift > 1)
#summary(rulesOwn)
```

### Step 3

Find out top 10 largest lift in ```rulesOwn```.

``` r=
rulesOwn_sort = sort(rulesOwn, by = "lift")
inspect(rulesOwn_sort[1:10])
#     lhs                                                          rhs       
#[1]  {Sex=1,MaritalStatus=2,EconomicActivity=8,SocialGrade=3}  => {Health=5}
#[2]  {Age=7,EconomicActivity=8}                                => {Health=5}
#[3]  {Age=7,EconomicActivity=8,HoursWorked=-9}                 => {Health=5}
#[4]  {Age=7,Student=2,EconomicActivity=8}                      => {Health=5}
#[5]  {Age=7,Student=2,EconomicActivity=8,HoursWorked=-9}       => {Health=5}
#[6]  {Age=7,EthnicGroup=1,EconomicActivity=8}                  => {Health=5}
#[7]  {Age=7,EthnicGroup=1,EconomicActivity=8,HoursWorked=-9}   => {Health=5}
#[8]  {Age=7,Student=2,EthnicGroup=1,EconomicActivity=8}        => {Health=5}
#[9]  {Age=7,CountryOfBirth=1,EthnicGroup=1,EconomicActivity=8} => {Health=5}
#[10] {Age=7,CountryOfBirth=1,EconomicActivity=8}               => {Health=5}
#    support      confidence  lift     count
#[1] 0.0001794162 0.2121849   16.62677 101  
#[2] 0.0002273789 0.2067851   16.20365 128  
#[3] 0.0002273789 0.2067851   16.20365 128  
#[4] 0.0002273789 0.2067851   16.20365 128  
#[5] 0.0002273789 0.2067851   16.20365 128  
#[6] 0.0001989565 0.2036364   15.95691 112  
#[7] 0.0001989565 0.2036364   15.95691 112  
#[8] 0.0001989565 0.2036364   15.95691 112  
#[9] 0.0001811926 0.2000000   15.67197 102  
#[10]0.0001811926 0.1976744   15.48973 102
```

### Result

1. We find that people with the ages between 65 and 74, ~~with the economically inactive caused by long-term sick or disabled~~, without being a student, with the ethnic group of white, with the country of birth in UK instead of Non UK are the most likely rules of being unhealthy.

2. We need to discard the data with the economically inactive caused by long-term sick or disabled, since it is relatively same as the condition of "Health" is bad.

---
