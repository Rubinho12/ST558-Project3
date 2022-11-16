Project3 README
================
Ruben Sowah, Zhiyuan Yang
2022-11-14

# Purpose of the repository

This repository is created to complete the third project of a data
science class (ST558). The goal of this project is to read in an [online
news popularity data
set](https://archive.ics.uci.edu/ml/datasets/Online+News+Popularity),
perform variable selections, split the data into a train and test sets,
fit some machine learning models(linear regression, random forest,
boosted tree models) on the train set, predict the number of shares of
the articles on the test set, and see which model performs the best. The
comparison of the models as well as the whole project are automated
across six different channels.

# List of R Packages used

The following packages are used in completing the project

``` r
tidyverse  
caret  
Metrics  
ggplot2  
readr 
tibble
rsample  
randomForest  
rmarkdown  
```

# README render code

``` r
rmarkdown::render("Project3_README.Rmd", 
          output_format = "github_document",
          output_file = "README.md",
          output_options = list(
            html_preview = FALSE))
```

# Links to the generated analyses:

- [Entertainment channel analysis](data_channel_is_entertainment.md).  
- [Lifestyle channel analysis](data_channel_is_lifestyle.md).
- [Business channel analysis](data_channel_is_bus.md).
- [Social media channel analysis](data_channel_is_socmed.md).
- [Technology channel analysis](data_channel_is_tech.md).
- [World channel analysis](data_channel_is_world.md).

# Code used to create the analyses

- **Set up parameters for the automation code, print the reports**

``` r
channels <- c("data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world")

## Files
output_file <- paste0(channels,".md")

## Create a list for each channel with just channel name parameter
 params = lapply(channels, FUN = function(x){
 
   return(list(chan = x))

})
## Put into a tibble
reports = tibble(channels, output_file, params);reports
```

    ## # A tibble: 6 x 3
    ##   channels                      output_file                      params      
    ##   <chr>                         <chr>                            <list>      
    ## 1 data_channel_is_lifestyle     data_channel_is_lifestyle.md     <named list>
    ## 2 data_channel_is_entertainment data_channel_is_entertainment.md <named list>
    ## 3 data_channel_is_bus           data_channel_is_bus.md           <named list>
    ## 4 data_channel_is_socmed        data_channel_is_socmed.md        <named list>
    ## 5 data_channel_is_tech          data_channel_is_tech.md          <named list>
    ## 6 data_channel_is_world         data_channel_is_world.md         <named list>

- **Automation of the code**

``` r
## Automation
apply(reports, MARGIN = 1, FUN = function(x){

  rmarkdown::render(input = "Project3.Rmd",

                    output_format = "github_document",

                    output_file = x[[2]],

                    params = x[[3]],

                    output_options = list(html_preview = FALSE))

})
```
