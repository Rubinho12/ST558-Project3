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
    ##         6111         1235

``` r
# Two ways table of articles published on weekend and on Saturday or neither.
table(newsdata$is_weekend, newsdata$weekday_is_saturday)
```

    ##                
    ##                 Not Saturday It is Saturday
    ##   Not a weekend         6425              0
    ##   Weekend                396            525

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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-5-1.png)<!-- -->

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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-6-1.png)<!-- -->

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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-7-1.png)<!-- -->

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
    ## -15315  -1949  -1161    113 655373 
    ## 
    ## Coefficients:
    ##                                     Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                          3165.94     144.95  21.842  < 2e-16 ***
    ## n_tokens_content                      708.42     193.50   3.661 0.000254 ***
    ## num_hrefs                             655.24     176.11   3.721 0.000201 ***
    ## num_imgs                             -495.61     171.70  -2.887 0.003911 ** 
    ## num_videos                            188.76     146.40   1.289 0.197314    
    ## `weekday_is_mondayIt is Monday`       -98.42     147.63  -0.667 0.505024    
    ## `weekday_is_saturdayIt is Saturday`  -152.62     212.47  -0.718 0.472595    
    ## is_weekendWeekend                     294.73     214.10   1.377 0.168705    
    ## global_rate_positive_words           -386.55     149.11  -2.592 0.009560 ** 
    ## global_rate_negative_words             83.73     152.18   0.550 0.582225    
    ## avg_positive_polarity                 -39.44     148.74  -0.265 0.790904    
    ## avg_negative_polarity                 -87.49     152.11  -0.575 0.565189    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10390 on 5130 degrees of freedom
    ## Multiple R-squared:  0.01243,    Adjusted R-squared:  0.01031 
    ## F-statistic: 5.868 on 11 and 5130 DF,  p-value: 1.522e-09

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
    ##          6425           921

***Comments:*** Based on the contingency table, we can see how many
articles are published on weekend. 0 means articles are not published on
weekend. 1 means articles are published on weekend.

``` r
# Create contingency table of whether the article was published on the Saturday
table(newsdata$weekday_is_saturday)
```

    ## 
    ##   Not Saturday It is Saturday 
    ##           6821            525

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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-12-1.png)<!-- -->
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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

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

![](data_channel_is_tech_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

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
    ## -31911  -1944  -1000    408 640721 
    ## 
    ## Coefficients:
    ##                                                         Estimate Std. Error t value Pr(>|t|)    
    ## (Intercept)                                              3165.94     143.76  22.023  < 2e-16 ***
    ## n_tokens_content                                         4042.58    1634.83   2.473 0.013439 *  
    ## num_hrefs                                                2060.39    1313.52   1.569 0.116804    
    ## num_imgs                                                -2937.74    1126.50  -2.608 0.009138 ** 
    ## num_videos                                               3485.03    1677.78   2.077 0.037835 *  
    ## global_rate_positive_words                                164.27     632.83   0.260 0.795203    
    ## global_rate_negative_words                               -145.42     777.83  -0.187 0.851703    
    ## avg_positive_polarity                                     107.50     391.53   0.275 0.783671    
    ## avg_negative_polarity                                     373.09     764.11   0.488 0.625378    
    ## `weekday_is_mondayIt is Monday`                          -120.96     146.82  -0.824 0.410062    
    ## `weekday_is_saturdayIt is Saturday`                      -175.82     211.63  -0.831 0.406147    
    ## is_weekendWeekend                                         303.39     213.50   1.421 0.155362    
    ## `n_tokens_content:num_hrefs`                              197.19     295.53   0.667 0.504659    
    ## `n_tokens_content:num_imgs`                              -868.95     396.44  -2.192 0.028431 *  
    ## `n_tokens_content:num_videos`                             414.60     457.38   0.906 0.364730    
    ## `n_tokens_content:global_rate_positive_words`           -6372.39     920.40  -6.923 4.95e-12 ***
    ## `n_tokens_content:global_rate_negative_words`            -261.02     608.12  -0.429 0.667777    
    ## `n_tokens_content:avg_positive_polarity`                 -144.41    1550.81  -0.093 0.925811    
    ## `n_tokens_content:avg_negative_polarity`                -3083.34     846.23  -3.644 0.000271 ***
    ## `num_hrefs:num_imgs`                                        9.41     327.36   0.029 0.977068    
    ## `num_hrefs:num_videos`                                   -536.70     288.95  -1.857 0.063308 .  
    ## `num_hrefs:global_rate_positive_words`                   -564.32     735.50  -0.767 0.442956    
    ## `num_hrefs:global_rate_negative_words`                    448.42     460.69   0.973 0.330419    
    ## `num_hrefs:avg_positive_polarity`                        -863.18    1241.13  -0.695 0.486784    
    ## `num_hrefs:avg_negative_polarity`                         718.05     675.83   1.062 0.288070    
    ## `num_imgs:num_videos`                                      84.40     183.81   0.459 0.646132    
    ## `num_imgs:global_rate_positive_words`                    3146.83     656.75   4.792 1.70e-06 ***
    ## `num_imgs:global_rate_negative_words`                    -132.52     413.65  -0.320 0.748707    
    ## `num_imgs:avg_positive_polarity`                         1021.38    1042.05   0.980 0.327047    
    ## `num_imgs:avg_negative_polarity`                          612.56     581.74   1.053 0.292400    
    ## `num_videos:global_rate_positive_words`                 -1463.82     952.54  -1.537 0.124415    
    ## `num_videos:global_rate_negative_words`                   -79.16     539.86  -0.147 0.883427    
    ## `num_videos:avg_positive_polarity`                       -550.43    1404.37  -0.392 0.695119    
    ## `num_videos:avg_negative_polarity`                       1298.46     813.04   1.597 0.110317    
    ## `global_rate_positive_words:global_rate_negative_words`   355.43     530.84   0.670 0.503171    
    ## `global_rate_positive_words:avg_positive_polarity`        477.10     710.71   0.671 0.502061    
    ## `global_rate_positive_words:avg_negative_polarity`       -382.36     538.54  -0.710 0.477740    
    ## `global_rate_negative_words:avg_positive_polarity`       -236.20     705.68  -0.335 0.737853    
    ## `global_rate_negative_words:avg_negative_polarity`       -167.71     428.26  -0.392 0.695369    
    ## `avg_positive_polarity:avg_negative_polarity`             427.39     747.86   0.571 0.567695    
    ## ---
    ## Signif. codes:  0 '***' 0.001 '**' 0.01 '*' 0.05 '.' 0.1 ' ' 1
    ## 
    ## Residual standard error: 10310 on 5102 degrees of freedom
    ## Multiple R-squared:  0.03388,    Adjusted R-squared:  0.02649 
    ## F-statistic: 4.587 on 39 and 5102 DF,  p-value: < 2.2e-16

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
    ## 5142 samples
    ##   11 predictor
    ## 
    ## Pre-processing: centered (11), scaled (11) 
    ## Resampling: Cross-Validated (5 fold, repeated 3 times) 
    ## Summary of sample sizes: 4114, 4114, 4114, 4113, 4113, 4113, ... 
    ## Resampling results across tuning parameters:
    ## 
    ##   interaction.depth  n.trees  RMSE      Rsquared     MAE     
    ##   1                   25      8356.012  0.001212769  2544.168
    ##   1                   50      8360.892  0.001544441  2541.481
    ##   1                  100      8409.917  0.001915429  2552.078
    ##   1                  150      8447.689  0.001884163  2536.977
    ##   1                  200      8535.646  0.001929508  2551.551
    ##   1                  250      8594.585  0.001952389  2548.954
    ##   2                   25      8392.407  0.003793816  2538.841
    ##   2                   50      8498.208  0.004027646  2542.948
    ##   2                  100      8615.626  0.004100214  2548.712
    ##   2                  150      8810.955  0.003950029  2572.661
    ##   2                  200      9041.058  0.003966601  2593.054
    ##   2                  250      9235.221  0.003913300  2612.704
    ##   3                   25      8456.847  0.004340232  2543.172
    ##   3                   50      8593.899  0.004781457  2556.822
    ##   3                  100      8714.125  0.005306608  2552.902
    ##   3                  150      8939.313  0.005201644  2571.438
    ##   3                  200      9236.524  0.004946659  2601.764
    ##   3                  250      9367.857  0.005203850  2616.234
    ##   4                   25      8351.029  0.005728507  2513.804
    ##   4                   50      8441.663  0.006522541  2529.074
    ##   4                  100      8607.237  0.006721397  2561.682
    ##   4                  150      8904.965  0.005170608  2602.356
    ##   4                  200      9210.167  0.005187349  2632.835
    ##   4                  250      9432.245  0.004606660  2668.988
    ##   5                   25      8387.974  0.005972804  2529.097
    ##   5                   50      8551.972  0.005833975  2565.491
    ##   5                  100      8713.805  0.006188427  2590.800
    ##   5                  150      8935.767  0.005476973  2630.203
    ##   5                  200      9074.197  0.005106365  2649.056
    ##   5                  250      9357.322  0.005215828  2695.812
    ## 
    ## Tuning parameter 'shrinkage' was held constant at a value of 0.1
    ## Tuning parameter 'n.minobsinnode'
    ##  was held constant at a value of 10
    ## RMSE was used to select the optimal model using the smallest value.
    ## The final values used for the model were n.trees = 25, interaction.depth = 4, shrinkage = 0.1
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
