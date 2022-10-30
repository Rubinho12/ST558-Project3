Project3
================
Ruben Sowah
2022-10-30

-   <a href="#introduction" id="toc-introduction">Introduction</a>
-   <a href="#load-packages" id="toc-load-packages">Load packages</a>
-   <a href="#read-in-the-data" id="toc-read-in-the-data">Read in the
    data</a>
-   <a href="#summarizations" id="toc-summarizations">Summarizations</a>

# Introduction

# Load packages

``` r
library(tidyverse)
library(caret)
library(Metrics)
library(ggplot2)
library(readr)
```

# Read in the data

``` r
## Read and get an overview of the data
data <- read_csv("OnlineNewsPopularity.csv")
head(data)
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
data <- data %>% 
        filter(data_channel_is_entertainment == 1) %>%
        select(n_tokens_content,num_hrefs,num_imgs, num_videos,weekday_is_saturday,is_weekend,global_rate_positive_words,global_rate_negative_words,avg_positive_polarity,avg_negative_polarity,shares) 

## Coerce the categorical variables into factor
data$weekday_is_saturday <- as.factor(data$weekday_is_saturday)
data$is_weekend <- as.factor(data$is_weekend)


print(data, width = 100, n = 10)
```

    ## # A tibble: 7,057 x 11
    ##    n_tokens_content num_hrefs num_imgs num_videos weekday_is_saturday is_weekend
    ##               <dbl>     <dbl>    <dbl>      <dbl> <fct>               <fct>     
    ##  1              219         4        1          0 0                   0         
    ##  2              531         9        1          0 0                   0         
    ##  3              194         4        0          1 0                   0         
    ##  4              161         5        0          6 0                   0         
    ##  5              454         5        1          0 0                   0         
    ##  6              177         4        1          0 0                   0         
    ##  7              356         3       12          1 0                   0         
    ##  8              281         5        1          0 0                   0         
    ##  9              909         3        1          1 0                   0         
    ## 10              413         6       13          0 0                   0         
    ##    global_rate_positive_words global_rate_negative_words avg_po~1 avg_n~2 shares
    ##                         <dbl>                      <dbl>    <dbl>   <dbl>  <dbl>
    ##  1                     0.0457                     0.0137    0.379  -0.35     593
    ##  2                     0.0414                     0.0207    0.386  -0.370   1200
    ##  3                     0.0567                     0         0.545   0       2100
    ##  4                     0.0497                     0.0186    0.427  -0.364   1200
    ##  5                     0.0441                     0.0132    0.363  -0.215   4600
    ##  6                     0.0678                     0.0113    0.417  -0.167   1200
    ##  7                     0.0618                     0.0140    0.359  -0.373    631
    ##  8                     0.0463                     0.0214    0.322  -0.278   1300
    ##  9                     0.0649                     0.0220    0.381  -0.258   1700
    ## 10                     0.0412                     0.0121    0.345  -0.408    455
    ## # ... with 7,047 more rows, and abbreviated variable names 1: avg_positive_polarity,
    ## #   2: avg_negative_polarity

``` r
levels(data$weekday_is_saturday)       
```

    ## [1] "0" "1"

# Summarizations
