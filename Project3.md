Project3
================
Ruben Sowah and Zhiyuan Yang
2022-10-30

-   <a href="#introduction" id="toc-introduction">Introduction</a>
-   <a href="#load-packages" id="toc-load-packages">Load packages</a>
-   <a href="#read-in-the-data" id="toc-read-in-the-data">Read in the
    data</a>
-   <a href="#first-group-members-summarizations"
    id="toc-first-group-members-summarizations">First group member’s
    summarizations</a>
-   <a href="#first-group-members-modeling"
    id="toc-first-group-members-modeling">First group member’s modeling</a>
-   <a href="#second-group-members-summarizations"
    id="toc-second-group-members-summarizations">Second group member’s
    summarizations</a>
-   <a href="#second-group-members-modeling"
    id="toc-second-group-members-modeling">Second group member’s
    modeling</a>
-   <a href="#comparison-of-the-four-models"
    id="toc-comparison-of-the-four-models">Comparison of the four models</a>
-   <a href="#automation" id="toc-automation">Automation</a>
-   <a href="#conclusion" id="toc-conclusion">Conclusion</a>

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
```

# Read in the data

``` r
## Read and get an overview of the data
newsdata <- read_csv("OnlineNewsPopularity.csv")
head(newsdata)
```

    ## # A tibble: 6 x 61
    ##   url    timed~1 n_tok~2 n_tok~3 n_uni~4 n_non~5 n_non~6 num_h~7 num_s~8 num_i~9
    ##   <chr>    <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>   <dbl>
    ## 1 http:~     731      12     219   0.664    1.00   0.815       4       2       1
    ## 2 http:~     731       9     255   0.605    1.00   0.792       3       1       1
    ## 3 http:~     731       9     211   0.575    1.00   0.664       3       1       1
    ## 4 http:~     731       9     531   0.504    1.00   0.666       9       0       1
    ## 5 http:~     731      13    1072   0.416    1.00   0.541      19      19      20
    ## 6 http:~     731      10     370   0.560    1.00   0.698       2       2       0
    ## # ... with 51 more variables: num_videos <dbl>, average_token_length <dbl>,
    ## #   num_keywords <dbl>, data_channel_is_lifestyle <dbl>,
    ## #   data_channel_is_entertainment <dbl>, data_channel_is_bus <dbl>,
    ## #   data_channel_is_socmed <dbl>, data_channel_is_tech <dbl>,
    ## #   data_channel_is_world <dbl>, kw_min_min <dbl>, kw_max_min <dbl>,
    ## #   kw_avg_min <dbl>, kw_min_max <dbl>, kw_max_max <dbl>, kw_avg_max <dbl>,
    ## #   kw_min_avg <dbl>, kw_max_avg <dbl>, kw_avg_avg <dbl>, ...

``` r
## Subset the data by the entertainment channel, and select our desired features
newsdata <- newsdata %>% 
        filter(data_channel_is_entertainment == 1) %>%
        select(n_tokens_content,num_hrefs,num_imgs, num_videos,weekday_is_monday,weekday_is_saturday,is_weekend,global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,shares) 


## Coerce the categorical variables into factor
newsdata$weekday_is_monday <- factor(newsdata$weekday_is_monday, levels = c(0,1), labels = c('Not Monday', 'It is Monday'))

newsdata$weekday_is_saturday <- factor(newsdata$weekday_is_saturday, levels = c(0,1), labels = c('Not Saturday', 'It is Saturday'))

newsdata$is_weekend <- factor(newsdata$is_weekend, levels = c(0,1), labels = c('Not a weekend', 'Weekend'))

## View data
print(newsdata, width = 100, n = 10)
```

    ## # A tibble: 7,057 x 12
    ##    n_tokens_content num_hrefs num_imgs num_videos weekday_is_monday
    ##               <dbl>     <dbl>    <dbl>      <dbl> <fct>            
    ##  1              219         4        1          0 It is Monday     
    ##  2              531         9        1          0 It is Monday     
    ##  3              194         4        0          1 It is Monday     
    ##  4              161         5        0          6 It is Monday     
    ##  5              454         5        1          0 It is Monday     
    ##  6              177         4        1          0 It is Monday     
    ##  7              356         3       12          1 It is Monday     
    ##  8              281         5        1          0 Not Monday       
    ##  9              909         3        1          1 Not Monday       
    ## 10              413         6       13          0 Not Monday       
    ##    weekday_is_saturday is_weekend    global_rat~1 globa~2 avg_p~3 avg_n~4 shares
    ##    <fct>               <fct>                <dbl>   <dbl>   <dbl>   <dbl>  <dbl>
    ##  1 Not Saturday        Not a weekend       0.0457  0.0137   0.379  -0.35     593
    ##  2 Not Saturday        Not a weekend       0.0414  0.0207   0.386  -0.370   1200
    ##  3 Not Saturday        Not a weekend       0.0567  0        0.545   0       2100
    ##  4 Not Saturday        Not a weekend       0.0497  0.0186   0.427  -0.364   1200
    ##  5 Not Saturday        Not a weekend       0.0441  0.0132   0.363  -0.215   4600
    ##  6 Not Saturday        Not a weekend       0.0678  0.0113   0.417  -0.167   1200
    ##  7 Not Saturday        Not a weekend       0.0618  0.0140   0.359  -0.373    631
    ##  8 Not Saturday        Not a weekend       0.0463  0.0214   0.322  -0.278   1300
    ##  9 Not Saturday        Not a weekend       0.0649  0.0220   0.381  -0.258   1700
    ## 10 Not Saturday        Not a weekend       0.0412  0.0121   0.345  -0.408    455
    ## # ... with 7,047 more rows, and abbreviated variable names 1: global_rate_positive_words,
    ## #   2: global_rate_negative_words, 3: avg_positive_polarity, 4: avg_negative_polarity

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
```

    ## # A tibble: 1 x 5
    ##   tokens.avg image.avg vids.avg pos.words.dev links.var
    ##        <dbl>     <dbl>    <dbl>         <dbl>     <dbl>
    ## 1       607.      6.32     2.55        0.0169      167.

``` r
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

-   From the numerical summaries, the results show that there is an
    average of 607 words in the content, an average of 6 images and 3
    videos. The standard deviation of the positive words from the mean
    is 0.0169 and the number of links varies by a average amount of 167.

-   The one way contingency table tells us that the number of articles
    published on Monday is less compared to the number of articles that
    is not published on Monday, 1358 versus 5699.

-   From the two ways contingency tables, 536 articles are published
    during the weekend, but it is not on Saturday. The amount of
    articles published on Saturday is 380. A total number of 6141
    articles are not published during the weekend.

#### <u>2) Graphs</u>

-   **Scatter plot of the rate of positive words in the content and the
    number of shares**

``` r
g <- ggplot(newsdata, aes(x = global_rate_positive_words, y = shares))
g + geom_point(color = 'blue')+
  labs(title = 'Rate of positive words vs Number of shares')
```

![](Project3_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

A scatter plot is used to visualize the relation between two numeric
variables. A strong positive relationship between the rate of positive
words and the number of shares will show a linear upward trend with the
data points closed to each other. This means that the number of shares
grows as the number of positive words increases.

A negative relationship between the two variables is shown by a downward
trend that tells us that people share less contents that have lots of
positive words.

-   **Density plot**

``` r
g <- ggplot(newsdata, aes(x = global_rate_negative_words)) 
g + geom_density(kernel ='gaussian', color = 'red', size = 2)+
  labs(title = 'Density plot  of the rate of negative words in the article')
```

![](Project3_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

A density plot can tell us about the distribution of a certain feature
or the whole data. Here, we plot the density of the rate of negative
words. A right skewed plot is an indication that there are quite more
negative words in the article. A left skewed plot indicates that there
are not much of negative words in the article. A symmetric plot tells us
that the amount of negative words in the article is normally
distributed, about average.

<br>

-   **Dotplot**

``` r
g <- ggplot(newsdata, aes(x = is_weekend, y = shares)) 
g + geom_dotplot(binaxis = "y", stackdir = 'center', color = 'magenta', dotsize = 1.2)+
  labs(title = 'Dotplot of the number of articles shared vs the week of the day')
```

![](Project3_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

-   Similarly to a boxplot, dotplots can be used to visualize the five
    number summary of a numeric data. Here , we are trying to see
    graphically the number of contents shared during the weekday and the
    weekend. We would expect the minimum number to be 0, since a the
    least amount of contents that can be shared can’t go below 0.

-   A greater number of points, for example in the ‘Not weekend’ group
    states that more articles are shared during the week days compared
    the weekend. The opposite would mean that contents are shared more
    during the weekend.

-   Points that are far away from the rest indicates possible outliers.

# First group member’s modeling

Here the data will be split into two, a training set and a testing set.
Two different models will be fit on the training set , then later be
evaluated on the test set. The two models that will be fit are a
**linear regression model** and a **random forest model**, using
cross-validation.

-   **What is linear regression about ?**

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

-   **What is random forest about ?**

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

The data now will be split into a train and test sets, and a linear
regression model will be fit on the train set. The train set will be 70
percent of the whole data and the remaining 30 % will be the test set.

``` r
set.seed(12)

## Using the rsample package, create a training an test set (70/30)
index <- initial_split(newsdata, prop = 0.7)
train.set <- training(index)
test.set <- testing(index)

## Check the dimensions of the sets
dim(train.set)
```

    ## [1] 4939   12

``` r
dim(test.set)
```

    ## [1] 2118   12

``` r
## Fit a linear regression model
regmod <- lm(shares ~. , data = train.set)
summary(regmod)
```

    ## 
    ## Call:
    ## lm(formula = shares ~ ., data = train.set)
    ## 
    ## Residuals:
    ##    Min     1Q Median     3Q    Max 
    ##  -4650  -2135  -1672   -844 207803 
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                        2.163e+03  4.583e+02   4.719 2.44e-06 ***
    ## n_tokens_content                  -6.383e-02  2.620e-01  -0.244  0.80748    
    ## num_hrefs                          1.470e+01  9.752e+00   1.507  0.13182    
    ## num_imgs                           1.316e+01  1.122e+01   1.173  0.24076    
    ## num_videos                         1.204e+01  1.965e+01   0.613  0.54023    
    ## weekday_is_mondayIt is Monday      1.595e+02  2.883e+02   0.553  0.58001    
    ## weekday_is_saturdayIt is Saturday -4.801e+02  6.178e+02  -0.777  0.43712    
    ## is_weekendWeekend                  1.251e+03  4.234e+02   2.954  0.00315 ** 
    ## global_rate_positive_words        -8.560e+03  7.215e+03  -1.186  0.23551    
    ## global_rate_negative_words        -9.847e+03  9.912e+03  -0.994  0.32051    
    ## avg_positive_polarity              1.985e+03  1.156e+03   1.717  0.08598 .  
    ## avg_negative_polarity             -8.386e+02  9.667e+02  -0.867  0.38576    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 7793 on 4927 degrees of freedom
    ## Multiple R-squared:  0.004994,   Adjusted R-squared:  0.002773 
    ## F-statistic: 2.248 on 11 and 4927 DF,  p-value: 0.01011

``` r
train.set
```

    ## # A tibble: 4,939 x 12
    ##    n_tokens_co~1 num_h~2 num_i~3 num_v~4 weekd~5 weekd~6 is_we~7 globa~8 globa~9
    ##            <dbl>   <dbl>   <dbl>   <dbl> <fct>   <fct>   <fct>     <dbl>   <dbl>
    ##  1           643       6       1       0 Not Mo~ Not Sa~ Not a ~  0.0544 0.0358 
    ##  2           465       2       2       1 Not Mo~ Not Sa~ Not a ~  0.0366 0.0323 
    ##  3          1295      50       8       2 Not Mo~ Not Sa~ Not a ~  0.0548 0.0139 
    ##  4           685      20       1       0 Not Mo~ Not Sa~ Not a ~  0.0263 0.0102 
    ##  5           980       5       0      27 Not Mo~ Not Sa~ Not a ~  0.0571 0.0582 
    ##  6           639      21       1       0 Not Mo~ It is ~ Weekend  0.0313 0.00939
    ##  7           749      25      11       2 Not Mo~ It is ~ Weekend  0.0374 0.00534
    ##  8           152       4       0       1 It is ~ Not Sa~ Not a ~  0.0724 0.0132 
    ##  9           424      22       0      12 Not Mo~ Not Sa~ Not a ~  0.0283 0.00943
    ## 10           252       3       0       1 Not Mo~ Not Sa~ Not a ~  0.0397 0.0198 
    ## # ... with 4,929 more rows, 3 more variables: avg_positive_polarity <dbl>,
    ## #   avg_negative_polarity <dbl>, shares <dbl>, and abbreviated variable names
    ## #   1: n_tokens_content, 2: num_hrefs, 3: num_imgs, 4: num_videos,
    ## #   5: weekday_is_monday, 6: weekday_is_saturday, 7: is_weekend,
    ## #   8: global_rate_positive_words, 9: global_rate_negative_words

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
```

    ## Random Forest 
    ## 
    ## 4939 samples
    ##   11 predictor
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Cross-Validated (5 fold) 
    ## Summary of sample sizes: 3953, 3950, 3952, 3951, 3950 
    ## Resampling results across tuning parameters:
    ## 
    ##   mtry  RMSE      Rsquared     MAE     
    ##    1    7586.019  0.004854547  2978.319
    ##    2    7668.107  0.006635436  3083.203
    ##    3    7731.411  0.007241371  3153.010
    ##    4    7797.593  0.007284751  3201.993
    ##    5    7829.618  0.006905499  3219.191
    ##    6    7868.161  0.006561780  3235.896
    ##    7    7888.915  0.007217257  3256.040
    ##    8    7907.136  0.006524527  3260.511
    ##    9    7963.117  0.006533922  3281.364
    ##   10    7997.740  0.006522236  3296.997
    ##   11    8011.162  0.006419175  3305.757
    ##   12    8010.231  0.006496982  3304.502
    ##   13    8048.652  0.005787055  3310.968
    ##   14    8008.497  0.006230320  3306.812
    ##   15    7989.358  0.006426498  3297.151
    ##   16    8031.663  0.006589208  3306.460
    ##   17    7985.681  0.007076473  3303.178
    ##   18    8031.762  0.006073337  3309.343
    ##   19    8004.879  0.006290055  3309.667
    ##   20    8046.403  0.006033799  3318.623
    ## 
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final value used for the model was mtry = 1.

``` r
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

![](Project3_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
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

![](Project3_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

***Comments:*** Based on this histogram, we can see the distribution of
the number of shares. If the peak of the graph lies to the left side of
the center, it means that most of articles have small number of shares.
If the peak of the graph lies to the right side of the center, it means
that most of articles have large number of shares. If we see a bell
shape, it means that the number of articles have large number of shares
is similar with the number of articles have small number of shares. The
No means the articles were published on weekend. The Yes means the
articles were published on weekend.

# Second group member’s modeling

# Comparison of the four models

``` r
## Predict the RF model on the test set 
forest.pred <- predict(forest.tuned, newdata = test.set)

## Get the accuracy of the RF model
forest.acc <- confusionMatrix(forest.pred, test.set$shares)
forest.acc
```

# Automation

# Conclusion
