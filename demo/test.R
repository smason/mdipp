source("scripts/analysis.R")
source("demo/datagen.R")

# draw a sample from our dirichlet process
base <- rdirichletprocess(100)
# draw a dataset from each of our datatypes and write out so MDI can see it
write.csv(create.normgam(base,        10),  "demo/normgam.csv")
write.csv(create.gaussianprocess(base,20),  "demo/gaussianprocess.csv")
write.csv(create.multinom(base, rep(3,10)), "demo/multinom.csv")
write.csv(create.bagofwords(base,100, 10),  "demo/bagwords.csv")

# run MDI
system("cd demo; ../mdi++ N normgam.csv GP gaussianprocess.csv M multinom.csv BW bagwords.csv > mcmc.csv")

# load our demo data back in (normally you won't be generating data in R!)
yg  <- loadDataGauss("demo/normgam.csv")
ygp <- loadDataGP("demo/gaussianprocess.csv")
ymn <- loadDataMultinom("demo/multinom.csv")
ybw <- loadDataBagOfWords("demo/bagwords.csv")

# put them in a list for easy access
datafiles <- list(yg,ygp,ymn,ybw)

# make sure data looks OK
par(mfrow=c(2,2))
for (y in datafiles) plot(y)

# load the MC output
mcmc <- readMdiMcmcOutput("demo/mcmc.csv")

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
