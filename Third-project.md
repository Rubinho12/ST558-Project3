README
================
Ruben Sowah
2022-11-14

``` r
channels <- c("data_channel_is_lifestyle", "data_channel_is_entertainment", "data_channel_is_bus", "data_channel_is_socmed", "data_channel_is_tech", "data_channel_is_world")



output_file <- paste0(channels,".md")

# Create a list for each channel with just channel name parameter
 params = lapply(channels, FUN = function(x){
 
   return(list(chan = x))

})

# Put into a data frame
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

``` r
# Automation

apply(reports, MARGIN = 1, FUN = function(x){

render(input = "Project3.Rmd",

                    #output_format = "github_document",

                    output_file = x[[2]],

                    params = x[[3]])
                    #params = list(chan = chan, name = name),

                    #output_options = list(html_preview = FALSE)

})
```

Here are the the reports that are generated for each day:

-   [Entertainment channel analysis](Project3.md).  
-   [Lifestyle channel analysis](data_channel_is_lifestyle.md).
-   [Business channel analysis](data_channel_is_bus.md).
-   [Social media channel analysis](data_channel_is_socmed.md).
-   [Technology channel analysis](data_channel_is_tech.md).
-   [World channel analysis](data_channel_is_world.md).