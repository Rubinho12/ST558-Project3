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


News_Data <- read_csv("C:\\Users\\zyang\\Desktop\\OnlineNewsPopularity\\OnlineNewsPopularity.csv")
# Subset the data to work on the data channel of interest

NewsData <- News_Data %>% filter(data_channel_is_entertainment == 1) %>% select(n_tokens_content, num_hrefs,
                                                                                    num_imgs, num_videos, is_weekend,
                                                                                    global_rate_positive_words, global_rate_negative_words,
                                                                                    avg_positive_polarity, avg_negative_polarity, weekday_is_saturday, shares)




NewsData$is_weekend <- as.factor(NewsData$is_weekend)

NewsData$weekday_is_saturday <- as.factor(NewsData$weekday_is_saturday)

str(NewsData)

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






#set the seed
set.seed(233)

#Split the data into a training and test set (70/30 split)
trainIndex <- createDataPartition(NewsData$shares, p = 0.70, list = FALSE)

# trainiing and testing subsets
sbikeTrain <- NewsData[trainIndex,]
sbikeTest <- NewsData[-trainIndex,]


sbikeTrain
sbikeTest