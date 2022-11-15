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
    ##         5699         1358

``` r
# Two ways table of articles published on weekend and on Saturday or neither.
table(newsdata$is_weekend, newsdata$weekday_is_saturday)
```

    ##                
    ##                 Not Saturday It is Saturday
    ##   Not a weekend         6141              0
    ##   Weekend                536            380

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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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
    ##  -4650  -2135  -1672   -844 207803 
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                          3008.83     110.88  27.135  < 2e-16 ***
    ## n_tokens_content                      -34.28     140.69  -0.244  0.80748    
    ## num_hrefs                             186.04     123.43   1.507  0.13182    
    ## num_imgs                              156.43     133.33   1.173  0.24076    
    ## num_videos                             74.41     121.48   0.613  0.54023    
    ## `weekday_is_mondayIt is Monday`        62.73     113.35   0.553  0.58001    
    ## `weekday_is_saturdayIt is Saturday`  -110.29     141.92  -0.777  0.43712    
    ## is_weekendWeekend                     424.50     143.72   2.954  0.00315 ** 
    ## global_rate_positive_words           -145.56     122.69  -1.186  0.23551    
    ## global_rate_negative_words           -125.44     126.26  -0.994  0.32051    
    ## avg_positive_polarity                 210.48     122.56   1.717  0.08598 .  
    ## avg_negative_polarity                -108.65     125.25  -0.867  0.38576    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7793 on 4927 degrees of freedom
    ## Multiple R-squared:  0.004994,   Adjusted R-squared:  0.002773 
    ## F-statistic: 2.248 on 11 and 4927 DF,  p-value: 0.01011

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
    ##          6141           916

***Comments:*** Based on the contingency table, we can see how many
articles are published on weekend. 0 means articles are not published on
weekend. 1 means articles are published on weekend.

``` r
# Create contingency table of whether the article was published on the Saturday
table(newsdata$weekday_is_saturday)
```

    ## 
    ##   Not Saturday It is Saturday 
    ##           6677            380

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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

![](data_channel_is_entertainment_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

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
    ##  -9133  -2171  -1544   -607 204565 
    ## 
    ## Coefficients:
    ##                                                           Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                              3008.8338   110.6968  27.181  < 2e-16 ***
    ## n_tokens_content                                         -452.7689  1001.1837  -0.452  0.65112    
    ## num_hrefs                                                -262.5246   861.0290  -0.305  0.76046    
    ## num_imgs                                                   -0.1806   505.1364   0.000  0.99971    
    ## num_videos                                               1514.0699  1023.5023   1.479  0.13912    
    ## global_rate_positive_words                               -663.4216   456.7086  -1.453  0.14639    
    ## global_rate_negative_words                                152.7237   673.2389   0.227  0.82055    
    ## avg_positive_polarity                                     508.5735   291.7776   1.743  0.08139 .  
    ## avg_negative_polarity                                     887.7558   534.4288   1.661  0.09675 .  
    ## `weekday_is_mondayIt is Monday`                            42.3394   113.5211   0.373  0.70919    
    ## `weekday_is_saturdayIt is Saturday`                       -78.9738   142.4488  -0.554  0.57933    
    ## is_weekendWeekend                                         388.9508   144.5000   2.692  0.00713 ** 
    ## `n_tokens_content:num_hrefs`                              -57.5909   270.3023  -0.213  0.83129    
    ## `n_tokens_content:num_imgs`                              -480.8278   227.0521  -2.118  0.03425 *  
    ## `n_tokens_content:num_videos`                            -598.5825   306.1223  -1.955  0.05060 .  
    ## `n_tokens_content:global_rate_positive_words`             -70.3935   555.1266  -0.127  0.89910    
    ## `n_tokens_content:global_rate_negative_words`            -978.9152   477.5070  -2.050  0.04041 *  
    ## `n_tokens_content:avg_positive_polarity`                 1365.4403   949.7236   1.438  0.15058    
    ## `n_tokens_content:avg_negative_polarity`                 -272.0893   696.6193  -0.391  0.69612    
    ## `num_hrefs:num_imgs`                                        0.3758   226.8739   0.002  0.99868    
    ## `num_hrefs:num_videos`                                   -267.0771   212.5814  -1.256  0.20905    
    ## `num_hrefs:global_rate_positive_words`                    -42.2805   434.7830  -0.097  0.92254    
    ## `num_hrefs:global_rate_negative_words`                    561.9000   316.1399   1.777  0.07557 .  
    ## `num_hrefs:avg_positive_polarity`                        -580.5220   585.3712  -0.992  0.32139    
    ## `num_hrefs:avg_negative_polarity`                        -662.4466   470.2632  -1.409  0.15900    
    ## `num_imgs:num_videos`                                      55.0378   157.0218   0.351  0.72597    
    ## `num_imgs:global_rate_positive_words`                     151.4075   320.0304   0.473  0.63616    
    ## `num_imgs:global_rate_negative_words`                     213.0509   283.5654   0.751  0.45249    
    ## `num_imgs:avg_positive_polarity`                         -346.8439   567.4218  -0.611  0.54105    
    ## `num_imgs:avg_negative_polarity`                         -413.8272   504.8737  -0.820  0.41245    
    ## `num_videos:global_rate_positive_words`                  -175.7914   602.5024  -0.292  0.77048    
    ## `num_videos:global_rate_negative_words`                  -101.1534   420.7206  -0.240  0.81001    
    ## `num_videos:avg_positive_polarity`                       -881.5733   841.0536  -1.048  0.29461    
    ## `num_videos:avg_negative_polarity`                       -365.3125   731.0376  -0.500  0.61730    
    ## `global_rate_positive_words:global_rate_negative_words`  -234.8895   452.2093  -0.519  0.60349    
    ## `global_rate_positive_words:avg_positive_polarity`        271.6954   462.2073   0.588  0.55668    
    ## `global_rate_positive_words:avg_negative_polarity`       -725.7470   421.7974  -1.721  0.08539 .  
    ## `global_rate_negative_words:avg_positive_polarity`       -907.6464   566.7522  -1.601  0.10933    
    ## `global_rate_negative_words:avg_negative_polarity`      -1082.6874   432.3514  -2.504  0.01231 *  
    ## `avg_positive_polarity:avg_negative_polarity`             203.6351   539.1400   0.378  0.70567    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7780 on 4899 degrees of freedom
    ## Multiple R-squared:  0.01398,    Adjusted R-squared:  0.006129 
    ## F-statistic: 1.781 on 39 and 4899 DF,  p-value: 0.002035

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
    ## 4939 samples
    ##   11 predictor
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 3951, 3951, 3950, 3953, 3951, 3951, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      7674.499  0.002050739  2987.937
    ##   1                   50      7673.313  0.002208481  2977.440
    ##   1                  100      7683.850  0.003181025  2981.274
    ##   1                  150      7669.517  0.004116853  2972.606
    ##   1                  200      7667.859  0.004542849  2980.651
    ##   1                  250      7663.040  0.005391222  2963.899
    ##   2                   25      7665.933  0.004239011  2976.448
    ##   2                   50      7671.026  0.005476362  2972.206
    ##   2                  100      7689.812  0.005519002  2983.511
    ##   2                  150      7701.333  0.005689865  2984.035
    ##   2                  200      7714.843  0.005132785  2999.384
    ##   2                  250      7726.844  0.005301718  3008.122
    ##   3                   25      7685.698  0.003292879  2985.982
    ##   3                   50      7708.280  0.004092520  2993.940
    ##   3                  100      7753.102  0.003630440  3017.964
    ##   3                  150      7803.863  0.002951750  3051.952
    ##   3                  200      7829.580  0.002878124  3074.407
    ##   3                  250      7881.279  0.002662101  3117.948
    ##   4                   25      7709.895  0.002579560  2993.667
    ##   4                   50      7753.952  0.002833271  3006.951
    ##   4                  100      7838.935  0.002100760  3055.129
    ##   4                  150      7896.931  0.001772223  3107.590
    ##   4                  200      7932.318  0.002088517  3151.646
    ##   4                  250      7975.967  0.002195975  3209.952
    ##   5                   25      7716.129  0.002661492  2992.111
    ##   5                   50      7784.849  0.002166497  3032.418
    ##   5                  100      7861.603  0.002104160  3079.518
    ##   5                  150      7931.262  0.001764055  3140.843
    ##   5                  200      7998.388  0.001459275  3221.424
    ##   5                  250      8051.507  0.001430277  3293.088
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode'
    ##  was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 250, interaction.depth = 1, shrinkage = 0.1
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