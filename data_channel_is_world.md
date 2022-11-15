Project3
================
Ruben Sowah,Zhiyuan Yang
2022-10-30

# Introduction

The dataset that we used is called Online News Popularity Data set. This
dataset concluded multiple features of articles published by Mashable in
past years. Our goal is to use different predictive models to predict
the number of shares in social networks. Our target variable is the
number of shares. The variable name of our target variable is called
shares. Thus, shares will be our dependent variable in our predictive
models. After our discussion, we both think the rate of unique words in
the content, the number of links, the number of images, the number of
videos, whether the article was published on the weekend, the rate of
positive words in the content, the rate of negative words in the
content, the average polarity of positive words, the average polarity of
negative words, whether the article published on Monday or on a Saturday
will affect the number of shares for article. Thus, we selected \*\*
n_tokens_content, num_hrefs, num_imgs, num_videos, is_weekend,
global_rate_positive_words, global_rate_negative_words,
avg_positive_polarity, avg_negative_polarity, weekday_is_monday,
weekday_is_saturday \*\* as our independent variables.

This dataset has six different channels, which are a lifestyle channel,
an entertainment channel, a bus channel, a social media channel, a tech
channel, and a world channel. We will subset the dataset based on
different channel types before we create our predictive models.

We will fit a random forest model and fit a boosted tree model. Both
models will be chosen using cross-validation. We will describe those in
more detail later.

# Load packages

``` r
library(tidyverse)
library(caret)
library(Metrics)
library(ggplot2)
library(readr)
library(corrplot)
library(knitr)
library(rsample)
library(randomForest)
library(rmarkdown)
library(tibble)
library(haven)
```

# Read in the data

``` r
## Read and get an overview of the data
newsdata <- read_csv("OnlineNewsPopularity.csv")
#newsdata <- read_csv("C:\\Users\\zyang\\Desktop\\OnlineNewsPopularity\\OnlineNewsPopularity.csv")
head(newsdata)

## Subset the data by the channels, and select our desired features
newsdata <- newsdata %>% 
        filter(!!rlang::sym(params$chan) == 1) %>%
        select(n_tokens_content,num_hrefs,num_imgs, num_videos,weekday_is_monday,weekday_is_saturday,is_weekend,global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,shares)#,data_channel_is_entertainment,data_channel_is_world) 
        
## Coerce the categorical variables into factor
newsdata$weekday_is_monday <- factor(newsdata$weekday_is_monday, levels = c(0,1), labels = c('Not Monday', 'It is Monday'))

newsdata$weekday_is_saturday <- factor(newsdata$weekday_is_saturday, levels = c(0,1), labels = c('Not Saturday', 'It is Saturday'))

newsdata$is_weekend <- factor(newsdata$is_weekend, levels = c(0,1), labels = c('Not a weekend', 'Weekend'))

## View data
print(newsdata, width = 100, n = 10)
```

# First group member’s summarizations

<br>

#### <u>1) Numerical summaries</u>

Here I will get some numerical summaries like the mean, the standard
deviation , the variance of some of the quantitative variables as well
as get the count of the categorical variables.

``` r
## Get the numerical summaries of some numeric features
num.summary <- newsdata %>%
          summarize(tokens.avg = mean(n_tokens_content), image.avg = mean(num_imgs), vids.avg = mean(num_videos), pos.words.dev = sd(global_rate_positive_words), links.var = var(num_hrefs))

num.summary

## Get contingency tables of the categorical features

# Count of the articles published and not on Monday
table(newsdata$weekday_is_monday)
```

    ## 
    ##   Not Monday It is Monday 
    ##         7071         1356

``` r
# Two ways table of articles published on weekend and on Saturday or neither.
table(newsdata$is_weekend, newsdata$weekday_is_saturday)
```

    ##                
    ##                 Not Saturday It is Saturday
    ##   Not a weekend         7341              0
    ##   Weekend                567            519

- From the numerical summaries, the results show that there is an
  average of 607 words in the content, an average of 6 images and 3
  videos. The standard deviation of the positive words from the mean is
  0.0169 and the number of links varies by a average amount of 167.

- The one way contingency table tells us that the number of articles
  published on Monday is less compared to the number of articles that is
  not published on Monday, 1358 versus 5699.

- From the two ways contingency tables, 536 articles are published
  during the weekend, but it is not on Saturday. The amount of articles
  published on Saturday is 380. A total number of 6141 articles are not
  published during the weekend.

#### <u>2) Graphs</u>

- **Scatter plot of the rate of positive words in the content and the
  number of shares**

``` r
g <- ggplot(newsdata, aes(x = global_rate_positive_words, y = shares))
g + geom_point(color = 'blue')+
  labs(title = 'Rate of positive words vs Number of shares')
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

A scatter plot is used to visualize the relation between two numeric
variables. A strong positive relationship between the rate of positive
words and the number of shares will show a linear upward trend with the
data points closed to each other. This means that the number of shares
grows as the number of positive words increases.

A negative relationship between the two variables is shown by a downward
trend that tells us that people share less contents that have lots of
positive words.

- **Density plot**

``` r
g <- ggplot(newsdata, aes(x = global_rate_negative_words)) 
g + geom_density(kernel ='gaussian', color = 'red', size = 2)+
  labs(title = 'Density plot  of the rate of negative words in the article')
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

A density plot can tell us about the distribution of a certain feature
or the whole data. Here, we plot the density of the rate of negative
words. A right skewed plot is an indication that there are quite more
negative words in the article. A left skewed plot indicates that there
are not much of negative words in the article. A symmetric plot tells us
that the amount of negative words in the article is normally
distributed, about average.

<br>

- **Dotplot**

``` r
g <- ggplot(newsdata, aes(x = is_weekend, y = shares)) 
g + geom_dotplot(binaxis = "y", stackdir = 'center', color = 'magenta', dotsize = 1.2)+
  labs(title = 'Dotplot of the number of articles shared vs the week of the day')
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

- Similarly to a boxplot, dotplots can be used to visualize the five
  number summary of a numeric data. Here , we are trying to see
  graphically the number of contents shared during the weekday and the
  weekend. We would expect the minimum number to be 0, since a the least
  amount of contents that can be shared can’t go below 0.

- A greater number of points, for example in the ‘Not weekend’ group
  states that more articles are shared during the week days compared the
  weekend. The opposite would mean that contents are shared more during
  the weekend.

- Points that are far away from the rest indicates possible outliers.

# First group member’s modeling

Here the data will be split into two, a training set and a testing set.
Two different models will be fit on the training set , then later be
evaluated on the test set. The two models that will be fit are a
**linear regression model** and a **random forest model**, using
cross-validation.

- **What is linear regression about ?**

Linear regression (LR) is the simplest form of a supervised machine
learning, where the data has both a single (simple linear regression) or
numerous predictors variables (multiple linear regression) denoted X’s
and an outcome or response variable denoted Y, that is quantitative.
Linear regression is used for either predicting the response variable or
to understand the relationship between the response and the predictors.
In the former case, we talk about prediction and in the latter, we talk
about inference.

Though a very simple approach , LR is widely used in practice and lots
of advanced models are a generalization of LR. With LR, one can seek to
understand if there is a relation between the response and the
predictors, and how strong that relationship is. Which predictors are
associated with the response, how accurately can one predicts the
response, is the relationship linear or non-linear, are the predictors
correlated? Those are some important questions one can answers with the
use of linear regression.

- **What is random forest about ?**

Random forest (RF) is supervised statistical machine learning algorithm
, constructed from decision trees, that is used in regression and
classification problems. RF is part of a general learning method called
*ensemble learning*. The idea of ensemble learning is to build a
prediction model by combining the strengths of a collection of simpler
base models, or in layman terms, an ensemble learning simply means
combining multiple models.

RF builds decision trees on different samples and takes their majority
vote for classification and average for regression. It is an extension
of another ensemble learning method called *Bagging or Bootstrap
Aggregation*. Bagging chooses a random sample from the data, and
generates different models from those samples called Bootstrap samples,
the sample is usually done with replacement.

Rf is an extension of Bagging in the sense that RF doesn’t use all the
predictors unlike Bagging. It uses a random subset of predictors for
each bootstrap sample, and the final output is based on the average or
majority ranking, in this way the problem of overfitting is also
avoided.

#### <u>**1) Fit a linear regression model**</u>

The data now will be split into a train and test sets, and a multiple
linear regression model will be fit on the train set. The train set will
be 70 percent of the whole data and the remaining 30 % will be the test
set.

``` r
## Set a seed for reproducible random numbers
set.seed(12)

## Using the rsample package, create a training an test set (70/30)
index <- initial_split(newsdata, prop = 0.7)
train.set <- training(index)
test.set <- testing(index)

## Fit a linear regression model
regmod <- train(shares ~. ,
                data = train.set,
                method = 'lm',
                preProcess = c('center','scale'),
                trControl = trainControl(method = 'cv', number = 5)
                )
summary(regmod)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -8951  -1450   -987   -277 281780 
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                          2335.72      82.57  28.289  < 2e-16 ***
    ## n_tokens_content                     -490.05      94.65  -5.177 2.33e-07 ***
    ## num_hrefs                             271.35      91.70   2.959  0.00310 ** 
    ## num_imgs                              725.38      88.30   8.215 2.59e-16 ***
    ## num_videos                            233.83      83.12   2.813  0.00492 ** 
    ## `weekday_is_mondayIt is Monday`       100.48      83.77   1.199  0.23043    
    ## `weekday_is_saturdayIt is Saturday`    54.42     109.98   0.495  0.62076    
    ## is_weekendWeekend                     158.75     111.03   1.430  0.15285    
    ## global_rate_positive_words            274.22      89.37   3.068  0.00216 ** 
    ## global_rate_negative_words            -29.11      87.41  -0.333  0.73915    
    ## avg_positive_polarity                -125.66      93.26  -1.347  0.17790    
    ## avg_negative_polarity                -169.73      88.69  -1.914  0.05570 .  
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6341 on 5886 degrees of freedom
    ## Multiple R-squared:  0.01818,    Adjusted R-squared:  0.01634 
    ## F-statistic: 9.908 on 11 and 5886 DF,  p-value: < 2.2e-16

#### <u>**2) Fit a random forest model**</u>

Here a random forest model will be fit on the train set using a
cross-validation with 5 folds. We will use the expand.grid() function to
select a range of parameters that will be tuned in our model. The
optimal parameter that minimizes th error will be chosen and the model
will be refit on the train set using that optimal parameter. We will
also center and scale the train data for a more accurate distribution of
the variables.

``` r
## Create a grid of tuning parameters
forestgrid <- expand.grid(mtry = c(1:20))

## Fit the random forest model
forestmod <- train(shares ~ . ,
                   data = train.set,
                   method = 'rf',
                   trControl = trainControl(method = 'cv', number= 5),
                   preProcess = c('center','scale'),
                   tuneGrid = forestgrid)
forestmod

## Get the optimal tuned parameter
mtry.opt <- forestmod$bestTune$mtry

## Refit the random forest model on the train set using the optimal tuned parameter
forest.tuned <-  train(shares ~ . ,
                   data = train.set,
                   method = 'rf',
                   trControl = trainControl(method = 'cv', number= 5),
                   preProcess = c('center','scale'),
                   tuneGrid = expand.grid(mtry = mtry.opt))
```

# Second group member’s summarizations

``` r
# Create contingency table of whether the article was published on the weekend
table(newsdata$is_weekend)
```

    ## 
    ## Not a weekend       Weekend 
    ##          7341          1086

***Comments:*** Based on the contingency table, we can see how many
articles are published on weekend. 0 means articles are not published on
weekend. 1 means articles are published on weekend.

``` r
# Create contingency table of whether the article was published on the Saturday
table(newsdata$weekday_is_saturday)
```

    ## 
    ##   Not Saturday It is Saturday 
    ##           7908            519

***Comments:*** Based on the contingency table, we can see how many
articles are published on Saturday. 0 means articles are not published
on Saturday. 1 means articles are published on Saturday.

``` r
# Create bar plot to see whether the article was published on the Saturday

ggplot(newsdata, aes(x=weekday_is_saturday))+
  geom_bar(aes(fill = "drv")) + 
  labs(y="Number of the Articles Were Published on the Saturday", 
       title= "Weekend published article's Bar plot")
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
***Comments:*** Based on the bar plot, we can see how many articles are
published on Saturday.

``` r
# Create histogram to see number of shares and whether the article was published on the Weekend

ggplot(data = newsdata, aes(x = shares))+ 
  geom_histogram(bins = 20, aes(fill = is_weekend)) +
  labs(x = "Number of Shares",
       y="Number of the Articles Were Published on the Weekend", 
       title = "Histogram of Shares that are Related to Weekend") +
       scale_fill_discrete(name = "Whether Weekend Published", 
                           labels = c("No", "Yes"))
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

***Comments:*** Based on this histogram, we can see the distribution of
the number of shares. If the peak of the graph lies to the left side of
the center, it means that most of articles have small number of shares.
If the peak of the graph lies to the right side of the center, it means
that most of articles have large number of shares. If we see a bell
shape, it means that the number of articles have large number of shares
is similar with the number of articles have small number of shares. The
No means the articles were published on weekend. The Yes means the
articles were published on weekend.

``` r
g <- ggplot(newsdata, aes(x = n_tokens_content, y = shares))
g + geom_point(color = 'green')+
  labs(title = 'number of tokens content vs Number of shares')
```

![](data_channel_is_world_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

***Comments:*** Based on this scatter plot, we can see how many points
plotted in the Cartesian plane. Each point represents the values of
number of shares and number of token content. The closer the data points
come to forming a straight line when plotted, it means that number of
shares and number of token content have stronger the relationship. If
the data points make a straight line going from near the origin out to
high y-values, variables will have a positive correlation.

# Second group member’s modeling

#### <u>Fit another linear regression model</u>

``` r
regmod2 <- train(shares~(n_tokens_content+num_hrefs+num_imgs+num_videos+global_rate_positive_words+
                           global_rate_negative_words+
                           avg_positive_polarity+avg_negative_polarity)^2+
                           weekday_is_monday+weekday_is_saturday+
                          is_weekend,
                 data = train.set,
                 method = "lm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 10))

summary(regmod2)
```

    ## 
    ## Call:
    ## lm(formula = .outcome ~ ., data = dat)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ## -17718  -1480   -857    -40 279207 
    ## 
    ## Coefficients:
    ##                                                          Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                              2335.721     81.807  28.551  < 2e-16 ***
    ## n_tokens_content                                         -816.170    700.880  -1.164 0.244272    
    ## num_hrefs                                               -1409.299    562.107  -2.507 0.012197 *  
    ## num_imgs                                                  470.070    203.890   2.306 0.021173 *  
    ## num_videos                                                356.069    520.405   0.684 0.493867    
    ## global_rate_positive_words                                963.158    374.163   2.574 0.010072 *  
    ## global_rate_negative_words                              -1288.144    414.462  -3.108 0.001893 ** 
    ## avg_positive_polarity                                    -374.955    251.078  -1.493 0.135391    
    ## avg_negative_polarity                                    -552.973    379.920  -1.455 0.145586    
    ## `weekday_is_mondayIt is Monday`                           122.027     83.169   1.467 0.142371    
    ## `weekday_is_saturdayIt is Saturday`                        78.646    109.200   0.720 0.471430    
    ## is_weekendWeekend                                         144.788    110.252   1.313 0.189152    
    ## `n_tokens_content:num_hrefs`                              459.606    211.734   2.171 0.029996 *  
    ## `n_tokens_content:num_imgs`                              -934.899    185.536  -5.039 4.82e-07 ***
    ## `n_tokens_content:num_videos`                             319.941    158.481   2.019 0.043555 *  
    ## `n_tokens_content:global_rate_positive_words`            1112.451    343.232   3.241 0.001197 ** 
    ## `n_tokens_content:global_rate_negative_words`             299.601    315.942   0.948 0.343026    
    ## `n_tokens_content:avg_positive_polarity`                  119.285    590.415   0.202 0.839895    
    ## `n_tokens_content:avg_negative_polarity`                 1100.738    389.604   2.825 0.004740 ** 
    ## `num_hrefs:num_imgs`                                      231.311    137.616   1.681 0.092845 .  
    ## `num_hrefs:num_videos`                                   -214.065    146.218  -1.464 0.143245    
    ## `num_hrefs:global_rate_positive_words`                  -1096.530    297.167  -3.690 0.000226 ***
    ## `num_hrefs:global_rate_negative_words`                    250.859    262.812   0.955 0.339861    
    ## `num_hrefs:avg_positive_polarity`                        1368.181    475.480   2.877 0.004023 ** 
    ## `num_hrefs:avg_negative_polarity`                        -911.413    297.266  -3.066 0.002179 ** 
    ## `num_imgs:num_videos`                                     -33.573    110.069  -0.305 0.760360    
    ## `num_imgs:global_rate_positive_words`                   -1031.085    228.461  -4.513 6.51e-06 ***
    ## `num_imgs:global_rate_negative_words`                     327.315    199.263   1.643 0.100513    
    ## `num_imgs:avg_positive_polarity`                          639.506    336.903   1.898 0.057720 .  
    ## `num_imgs:avg_negative_polarity`                         -983.839    262.997  -3.741 0.000185 ***
    ## `num_videos:global_rate_positive_words`                  -149.834    314.500  -0.476 0.633793    
    ## `num_videos:global_rate_negative_words`                    -9.507    212.957  -0.045 0.964393    
    ## `num_videos:avg_positive_polarity`                          1.266    383.893   0.003 0.997368    
    ## `num_videos:avg_negative_polarity`                         37.816    253.898   0.149 0.881603    
    ## `global_rate_positive_words:global_rate_negative_words`  -688.721    261.967  -2.629 0.008585 ** 
    ## `global_rate_positive_words:avg_positive_polarity`       -574.737    380.040  -1.512 0.130510    
    ## `global_rate_positive_words:avg_negative_polarity`       -649.949    276.273  -2.353 0.018678 *  
    ## `global_rate_negative_words:avg_positive_polarity`       1540.260    366.225   4.206 2.64e-05 ***
    ## `global_rate_negative_words:avg_negative_polarity`        -62.823    258.709  -0.243 0.808145    
    ## `avg_positive_polarity:avg_negative_polarity`            1046.480    391.201   2.675 0.007493 ** 
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 6283 on 5858 degrees of freedom
    ## Multiple R-squared:  0.04077,    Adjusted R-squared:  0.03438 
    ## F-statistic: 6.383 on 39 and 5858 DF,  p-value: < 2.2e-16

#### What is a boosted tree model?

The boosted tree model is a general approach that can be applied to
trees. Trees grown sequentially and each subsequent tree is grown on a
modified version of original data. When tree growing, the predictions
also are updated. Thus, it solves errors that created by previous
decision trees. Boosting transforms weak decision trees, which are weak
learners into strong learners. Boosting is an iterative process. Each
tree is dependent on the previous tree. For the procedure, we can
initialize predictions as 0, and Find the residuals
(observed-predicted), call the set of them r. And then we fit a tree
with splits (terminal nodes) treating the residuals as the response,
which they are for the first fit. After that, we can update predictions
and update residuals for new predictions and repeat B times.

#### <u>Fit a boosted tree model</u>

``` r
boosted_fit <- train(shares ~., data = train.set, method = "gbm",
                       trControl = trainControl(method = "repeatedcv", 
                                                number = 5, repeats = 3),
                       preProcess = c("center", "scale"),
                       tuneGrid = expand.grid(n.trees = c(25, 50, 100, 150, 200, 250),
                                              interaction.depth = 1:5,
                                              shrinkage = 0.1,
                                              n.minobsinnode = 10),
                       verbose = FALSE)
boosted_fit
```

    ## Stochastic Gradient Boosting 
    ## 
    ## 5898 samples
    ##   11 predictor
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4717, 4719, 4720, 4718, 4718, 4720, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared    MAE     
    ##   1                   25      6157.926  0.01375996  2006.508
    ##   1                   50      6155.225  0.01559562  2008.408
    ##   1                  100      6154.682  0.01718707  2005.645
    ##   1                  150      6154.175  0.01769996  1999.631
    ##   1                  200      6156.555  0.01814266  2004.891
    ##   1                  250      6158.796  0.01775001  2007.281
    ##   2                   25      6195.844  0.01179758  2020.914
    ##   2                   50      6227.717  0.01676979  2032.979
    ##   2                  100      6284.026  0.01608004  2073.724
    ##   2                  150      6333.479  0.01597658  2116.879
    ##   2                  200      6383.785  0.01413383  2164.997
    ##   2                  250      6412.410  0.01432535  2191.276
    ##   3                   25      6216.011  0.01291119  2029.090
    ##   3                   50      6239.836  0.01770847  2039.578
    ##   3                  100      6305.641  0.01723812  2094.460
    ##   3                  150      6327.205  0.01832131  2127.143
    ##   3                  200      6356.969  0.01813048  2156.550
    ##   3                  250      6368.253  0.02048127  2179.642
    ##   4                   25      6210.323  0.01418793  2029.765
    ##   4                   50      6235.620  0.01709399  2047.668
    ##   4                  100      6310.317  0.01793699  2107.509
    ##   4                  150      6337.143  0.01885752  2148.163
    ##   4                  200      6366.358  0.01976175  2177.986
    ##   4                  250      6390.539  0.01952023  2194.960
    ##   5                   25      6189.429  0.01885124  2017.225
    ##   5                   50      6235.593  0.01831891  2052.064
    ##   5                  100      6300.844  0.01939907  2112.091
    ##   5                  150      6344.427  0.01987160  2161.821
    ##   5                  200      6382.171  0.02131724  2194.497
    ##   5                  250      6421.083  0.02158055  2226.467
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode'
    ##  was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 150, interaction.depth = 1, shrinkage = 0.1
    ##  and n.minobsinnode = 10.

# Comparison of the four models

We will predict the four models fitted above on the test set and use the
postResample() function to get the test metrics. We are more concerned
about the root mean squared error (RMSE) as the measure of our models.

``` r
## Predict the multiple regression fit on the test set
regmod.pred <- predict(regmod, newdata = test.set)

## Get the RMSE of the regression model
regmod.rmse <- postResample(regmod.pred, test.set$shares)[1]

## Predict the RF model on the test set 
forest.pred <- predict(forest.tuned, newdata = test.set)

## Get the RMSE of the Random Forest model
forest.rmse <- postResample(forest.pred, test.set$shares)[1]

## Predict the second linear regression model on the test set
regmod2.pred <- predict(regmod2, newdata = test.set)

## Get the RMSE of the second linear regression model
regmod2.rmse <- postResample(regmod2.pred, test.set$shares)[1]

## Predict the boosted tree model on the test set
boosted.pred <- predict(boosted_fit, newdata = test.set)

## Get the RMSE of the boosted tree model
boosted.rmse <- postResample(boosted.pred, test.set$shares)[1]

## Combine the four RMSE in a table
data.frame(Regression = regmod.rmse,
           Forest = forest.rmse,
           Pol.regression = regmod2.rmse,
           Boosted.tree = boosted.rmse)
```

The **random forest** model has the lowest root mean squared error of
all four models, with a value of **7951.099**, hence is our winner
model.

# Automation

``` r
channels <- c("data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world")

## Files
output_file <- paste0(channels,".md")

## Create a list for each channel with just channel name parameter
 params = lapply(channels, FUN = function(x){
 
   return(list(chan = x))

})
## Put into a data frame
reports = tibble(channels, output_file, params);reports

## Automation

apply(reports, MARGIN = 1, FUN = function(x){

  rmarkdown::render(input = "Project3.Rmd",

                    output_format = "github_document",

                    output_file = x[[2]],

                    params = x[[3]],

                    output_options = list(html_preview = FALSE))

})
```
