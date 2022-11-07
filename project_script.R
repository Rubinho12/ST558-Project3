# Introduction section

You should have an introduction section that briefly describes the data and the variables you have to work
with (just discuss the ones you want to use). Your target variables is the shares variable.
You should also mention the purpose of your analysis and the methods you’ll use to model the response.
You’ll describe those in more detail later.

# read in the dataset

library(readr)
library(dplyr)
library(stats)
library(base)
library(corrplot)
library(knitr)
library(ggplot2)
library(lattice)
library(caret)
library(ggrastr)
library(GGally)
library(tidyverse)
library(caret)
library(Metrics)
library(ggplot2)
library(readr)
library(corrplot)
library(knitr)
library(rsample)



newsdata <- read_csv("C:\\Users\\zyang\\Desktop\\OnlineNewsPopularity\\OnlineNewsPopularity.csv")
# Subset the data to work on the data channel of interest

## Subset the data by the entertainment channel, and select our desired features
newsdata <- newsdata %>% 
  filter(data_channel_is_entertainment == 1) %>%
  select(n_tokens_content,num_hrefs,num_imgs, num_videos,weekday_is_monday,weekday_is_saturday,is_weekend,global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,shares) 


## Coerce the categorical variables into factor
newsdata$weekday_is_monday <- factor(newsdata$weekday_is_monday, levels = c(0,1), labels = c('Not Monday', 'It is Monday'))

newsdata$weekday_is_saturday <- factor(newsdata$weekday_is_saturday, levels = c(0,1), labels = c('Not Saturday', 'It is Saturday'))

newsdata$is_weekend <- factor(newsdata$is_weekend, levels = c(0,1), labels = c('Not a weekend', 'Weekend'))


#Three EDA


# Create contingency table of whether the article was published on the weekend
table(NewsData$is_weekend)


Based on the contingency table, we can see how many articles are published on weekend.
0 means articles are not published on weekend. 1 means articles are published on weekend.

# Create contingency table of whether the article was published on the Saturday
table(NewsData$weekday_is_saturday)

Based on the contingency table, we can see how many articles are published on Saturday.
0 means articles are not published on Saturday. 1 means articles are published on Saturday.


library(corrplot)
library(knitr)
library(ggplot2)

ggplot(NewsData, aes(x=weekday_is_saturday))+
  geom_bar(aes(fill = "drv")) + 
  labs(y="Number of the Articles Were Published on the Saturday", 
       title= "Weekend published article's Bar plot")

Based on the bar plot, we can see how many articles are published on Saturday.


# Create histogram to see number of shares and whether the article was published on the Weekend

ggplot(data = NewsData, aes(x = shares))+ 
  geom_histogram(bins = 20, aes(fill = is_weekend)) +
  labs(x = "Number of Shares",
       y="Number of the Articles Were Published on the Weekend", 
       title = "Histogram of Shares that are Related to Weekend") +
       scale_fill_discrete(name = "Whether Weekend Published", 
                           labels = c("No", "Yes"))


Based on this histogram, we can see the distribution of the number of shares. 

If the peak of the graph lies to the left side of the center, it means that 
most of articles have small number of shares. 
If the peak of the graph lies to the right side of the center, it means that most of articles have large number of shares. 
If we see a bell shape, it means that the number of articles have large number of shares is similar with 
the number of articles have small number of shares. 
The No means the articles were published on weekend. The Yes means
the articles were publshed on weekend.




set.seed(12)

## Using the rsample package, create a training an test set (70/30)
index <- initial_split(newsdata, prop = 0.7)
train.set <- training(index)
test.set <- testing(index)

## Check the dimensions of the sets
dim(train.set)
dim(test.set)


## another linear regression model fit

lm.fit2 <- train(shares~(n_tokens_content+num_hrefs+num_imgs+num_videos+global_rate_positive_words+
                           global_rate_negative_words+
                           avg_positive_polarity+avg_negative_polarity)^2+
                           weekday_is_monday+weekday_is_saturday+
                          is_weekend,
                 data = train.set,
                 method = "lm",
                 preProcess = c("center", "scale"),
                 trControl = trainControl(method = "cv", number = 10))

summary(lm.fit2)

#What is a boosted tree model.

The boosted tree model is a general approach that can be applied to trees.
Trees grown sequentially and each subsequent tree is grown on a modified version of original data.
When tree growing, the predictions also are updated.
Thus, it solves errors that created by previous decision trees. 
Boosting transforms weak decision trees, which are weak learners into strong learners.
Boosting is an iterative process. Each tree is dependent on the previous tree. 
For the procedure, we can initialize predictions as 0, and Find the residuals (observed-predicted), call the set of them r.
And then we fit a tree with splits (terminal nodes) treating the residuals as the
response, which they are for the first fit. After that, we can update predictions and update 
residuals for new predictions and repeat B times. 


#a boosted tree model. 


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


boostPred <- predict(boosted_fit, newdata = dplyr::select(test.set, -shares), n.trees = 5000)

boostRMSE <- sqrt(mean((boostPred-test.set$shares)^2))

boostRMSE