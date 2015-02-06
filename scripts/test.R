source("scripts/datagen.R")
source("scripts/analysis.R")

# draw a sample from our dirichlet process
base <- rdirichletprocess(100)
# draw a dataset from each of our datatypes and write out so MDI can see it
write.csv(create.normgam(base,        10),  "test/normgam.csv")
write.csv(create.gaussianprocess(base,20),  "test/gaussianprocess.csv")
write.csv(create.multinom(base, rep(3,10)), "test/multinom.csv")
write.csv(create.bagofwords(base,100, 10),  "test/bagwords.csv")

# run MDI
system("cd test; ../mdi++ N normgam.csv GP gaussianprocess.csv M multinom.csv BW bagwords.csv > mcmc.csv")

# load our test data back in (normally you won't be generating data in R!)
yg  <- loadDataGauss("test/normgam.csv")
ygp <- loadDataGP("test/gaussianprocess.csv")
ymn <- loadDataMultinom("test/multinom.csv")
ybw <- loadDataBagOfWords("test/bagwords.csv")

# put them in a list for easy access
datafiles <- list(yg,ygp,ymn,ybw)

# make sure data looks OK
par(mfrow=c(2,2))
for (y in datafiles) plot(y)

# load the MC output
mcmc <- readMdiMcmcOutput("test/mcmc.csv")

plotSingleMcmcSample(mcmc[1000,],datafiles)
plotSingleMcmcSample(mcmc[2000,],datafiles)
plotSingleMcmcSample(mcmc[4000,],datafiles)
plotSingleMcmcSample(mcmc[6000,],datafiles)
plotSingleMcmcSample(mcmc[8000,],datafiles)
plotSingleMcmcSample(mcmc[1000,],datafiles)

# plot allocation agreement between pairs of datasets over chain to ensure convergance
plotAllocationAgreement(tail(mcmc,2000))

# If all looks OK, just plot marginals 
plotAllocationAgreementHistogram(tail(mcmc,2000),datafiles)

# calculate the number of clusters to use below
nclust <- median(apply(getClustersOccupied(tail(mcmc,101)),2,median))

# generate Posteriour Similarity Matricies and a "consensus" (mean) PSM
cpsm <- generateConsensusPSM(tail(mcmc))
# plot the above
par(mar=c(1,2,1,1)); plotConsensusPSM(cpsm, datafiles, nclust,ann = TRUE)
