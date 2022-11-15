README
================
Ruben Sowah, Zhiyuan Yang
2022-11-14

# Automation code

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

# Reports for each channel:

-   [Entertainment channel analysis](Project3.md).  
-   [Lifestyle channel analysis](data_channel_is_lifestyle.md).
-   [Business channel analysis](data_channel_is_bus.md).
-   [Social media channel analysis](data_channel_is_socmed.md).
-   [Technology channel analysis](data_channel_is_tech.md).
-   [World channel analysis](data_channel_is_world.md).
