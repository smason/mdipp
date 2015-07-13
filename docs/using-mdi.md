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

There are two outputs from MDI, the output always present is the MC
output that targets the posterior distribution of cluster partitions.
When feature selection is turned on another file is generated
describing whether each feature is turned on or off.  For example, the
main MCMC output for clustering the above data as independent
Gaussians could look like:

    MassParameter_1,Dataset1_gene 1,Dataset1_gene 2,Dataset1_gene 3
    1,   1,1,2
    1.1, 1,1,1
    1.2, 2,2,1

The first column describes the Mass associated with the Dirichlet
Process of behind the clustering for the dataset.  This is followed by
gene 1 and 2 both in cluster 1 in the first sample, all genes were in
found to be compatible in the second sample.  When more than one
dataset is being clustered simultaneously there will be a series of
"Phi" variables describing the agreement between all pairwise
combinations of datasets.  More formally, given $N$ datasets each
containing $n$" items, the columns will be:

1. $N$ Mass Paramaters, followed by
2. $N*(N-1)/2$ Phis, and
3. $Nn$ cluster partition variables.

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
