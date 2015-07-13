# Using MDI++ #

## Example Analysis ##

The `demo` subdirectory includes some example data and scripts that
can be used to ensure that MDI++ is functioning correctly, as well as
providing a template for your own analysis.

TODO: more details

# File formats #

MDI++ reads and writes a variety of somewhat standard format CSV
files, these have allowed easy interoperability of data between MDI,
R, Python and Matlab.  The program assumes rectangular input format,
i.e. all rows have the same number of columns.  The for row contains
the feature names and the first column the names of the items, for
example:

    expression, idx, idy
	gene 1, 1.5, 2.3
	gene 2, 1.4, 4.0
	gene 3, 2.3, 0.7

defines a file with three genes as items, each having expression data
across two individuals.  The value of the topleft cell will be
ignored, but can be useful to remind you what the data is.  The format
of the numeric data here changes depending on the *data type*.

## Data Types ##

### `N`: Independent Gaussians ###

### `GP`:  Gaussian Process ###

### `M`: Multinomial ###

### `BW`: Bag of Words ###

## Analysis Scripts ##

One MDI++ has run the Monte Carlo output can be read into another
program for further analysis and plotting.  I have included various
scripts for analysing this posterior in R and there is a
iPython/Jupyter notebook demonstrating the use of these scripts.
