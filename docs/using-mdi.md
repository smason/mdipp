# Using MDI++ #

## Example Analysis ##

The `demo` subdirectory includes some example data and scripts that
can be used to ensure that MDI++ is functioning correctly, as well as
providing a template for your own analysis.

TODO: more details

## Analysis Scripts ##

Once the CSV files have been loaded various analyses and plots can be
generated to help you understand the clustering.  Inside R or RStudio,
plots can be generated

    > source('scripts/analysis.R')
    > dta <- loadDataFilesFromPath('demo/input.csv')
    > out <- readMdiMcmcOutput('demo/mcmc_01.csv')
    > cpsm <- generateConsensusPSM(out)
    > plotConsensusPSM(cpsm, dta, )
