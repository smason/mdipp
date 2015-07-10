library(compiler) # included in recent versions of R
library(mcclust)  # will need to install this the first time by running: install.packages("mcclust")

# convert a data.frame, as read by read.csv, into a matrix, taking the
# the first column and rownames.  changes some defaults so that
# identifiers don't get munged
read.csv.matrix <- function (path, as.is=TRUE, check.names=FALSE, ...) {
  df <- read.csv(path, as.is=as.is, check.names=check.names, ...)
  m <- as.matrix(df[,-1])
  colnames(m) <- colnames(df)[-1]
  rownames(m) <- df[[1]]
  m
}

# load all the "data files" from a given glob (i.e. /tmp/data-*.csv)
# extract the sample identifiers from filename
loadDataFilesFromPath <- function(glob) {
  names <- Sys.glob(glob)
  files <- lapply(names, read.csv.matrix)
  # filenames of the form "path/XXX.csv" will retrieve "XXX"
  names(files) <- sub("\\.csv$", "", basename(names))
  files
}

genPairStyles <- function(nitems) {
  a <- floor(sqrt(nitems))
  b <- ceiling(nitems/a)
  
  list(a=rep(1:a,length.out=nitems,each=b),
       b=rep(1:b,length.out=nitems))
}

loadDataGauss <- function(path, name=sub("\\.csv$", "", basename(path)),
                          col=NULL, lty=NULL, pch=NULL,
                          plot.type="p", colfn=rainbow) {
  y <- read.csv.matrix(path)

  nitems <- nrow(y)
  if(is.null(col) && is.null(lty) && is.null(pch)) {
    ps <- genPairStyles(nitems)
    
    col <- colfn(max(ps$a))[ps$a]
    pch <- ps$b
  } else {
    if(!is.null(col) && length(col) != nitems) stop("number of colors (@col) does not equal number of items")
    if(!is.null(lty) && length(lty) != nitems) stop("number of linetypes (@lty) does not equal number of items")
    if(!is.null(pch) && length(pch) != nitems) stop("number of plot-characters (@pch) does not equal number of items")
  }
  
  class(y) <- c("MdiData.Gauss","MdiScatterPlot",class(y))
  attributes(y) <- within(attributes(y),{
    name <- name
    time <- 1:ncol(y)
    type <- plot.type
    lty  <- lty
    col  <- col
    pch  <- pch
    xlim <- c(1,ncol(y))
    ylim <- range(y)
  })
  y
}

loadDataGP <- function(path, name=sub("\\.csv$", "", basename(path)),
                       col=NULL, lty=NULL, pch=NULL,
                       plot.type="b", colfn=rainbow) {
  y <- read.csv.matrix(path)
  
  nitems <- nrow(y)
  if(is.null(col) && is.null(lty) && is.null(pch)) {
    ps <- genPairStyles(nitems)
    
    col <- colfn(max(ps$a))[ps$a]
    pch <- ps$b
    lty <- ps$b
  } else {
    if(!is.null(col) && length(col) != nitems) stop("number of colors (@col) does not equal number of items")
    if(!is.null(lty) && length(lty) != nitems) stop("number of linetypes (@lty) does not equal number of items")
    if(!is.null(pch) && length(pch) != nitems) stop("number of plot-characters (@pch) does not equal number of items")
  }
  
  class(y) <- c("MdiData.Gp","MdiScatterPlot",class(y))
  attributes(y) <- within(attributes(y), {
    name <- name
    time <- as.numeric(colnames(y))
    type <- plot.type
    lty  <- lty
    col  <- col
    pch  <- pch
    xlim <- range(time)
    ylim <- range(y)
  })
  y
}

loadDataMultinom <- function(path, name=sub("\\.csv$", "", basename(path)),
                             colfn=rainbow) {
  y <- read.csv.matrix(path)
  class(y) <- c("MdiData.Mn","MdiRasterData",class(y))
  attributes(y) <- within(attributes(y), {
    name   <- name
    levels <- sort(unique(as.numeric(y)))
    col    <- colfn(length(levels))
    xlim   <- c(0,ncol(y))+0.5
    # flip Y-axis so the first item in the input file stays at the top of the plot
    ylim   <- c(nrow(y),0)+0.5
  })
  y
}

loadDataBagOfWords <- function(path, name=sub("\\.csv$", "", basename(path)),
                               breaks=NULL, col=grey.colors(64)) {
  y <- read.csv.matrix(path)
  if(is.null(breaks))
    breaks <- seq(min(y), max(y), len=length(col)+1)
  else {
    if (length(col)+1 != length(breaks))
      stop("Expecting one more break than colours, as per graphics::image()")
  }
  class(y) <- c("MdiData.Bow","MdiRasterData",class(y))
  attributes(y) <- within(attributes(y), {
    name   <- name
    breaks <- breaks
    col    <- col
    xlim   <- c(0,ncol(y))+0.5
    # flip Y-axis so the first item in the input file stays at the top of the plot
    ylim   <- c(nrow(y),0)+0.5
  })
  y
}

MdiLegendDraw <- function(y, ...) UseMethod("MdiLegendDraw", y)
# legend for scatter plots are items to colors/line/point styles
MdiLegendDraw.MdiScatterPlot <- function(y) {
  plot.new(); plot.window(0:1,0:1)
  legend("center", rownames(y), ncol=ceiling(nrow(y)/15),
         pch=attr(y,"pch"), lty=attr(y,"lty"), col=attr(y,"col"), bty="n")
}
# legends for raster based plots are colors to levels/values
MdiLegendDraw.MdiData.Mn <- function(y) {
  plot.new(); plot.window(0:1,0:1)
  legend("center", attr(y,"levels"), pch=20, bty="n")
}
MdiLegendDraw.MdiData.Bow <- function(y) {
  plot.new(); plot.window(0:1,0:1)
  
}

plot.MdiScatterPlot <- function(y, which=1:nrow(y),
                               xlim=attr(y,"xlim"), ylim=attr(y,"ylim"),
                               axes=TRUE) {
  if (length(axes) == 1) axes <- rep(axes,2)
  # initialise the plot area, axes and labels appropriately
  plot.new(); plot.window(xlim, ylim); box()
  for(i in seq_along(axes))
    axis(i, labels=axes[[i]], lwd=0, lwd.ticks=1)
  
  time <- attr(y,"time")
  type <- attr(y,"type")
  col <- attr(y,"col")
  lty <- attr(y,"lty")
  pch <- attr(y,"pch")
  
  # plot all the timeseries selected
  for (j in which)
    plot.xy(xy.coords(time, y[j,]), type, col=col[[j]],lty=lty[[j]],pch=pch[[j]])
}

as.raster.MdiData.Mn <- function(y) {
  img <- attr(y,"col")[unclass(factor(y,levels=attr(y,"levels")))]
  dim(img) <- dim(y)
  as.raster(img)
}

as.raster.MdiData.Bow <- function(y) {
  img <- attr(y,"col")[cut(y,attr(y,"breaks"),FALSE,TRUE)]
  dim(img) <- dim(y)
  as.raster(img)
}

plot.MdiRasterData <- function (y, which=1:nrow(y),
                             xlim=attr(y,"xlim"), ylim=attr(y,"ylim"),
                             axes=TRUE, ...) {
  # convert the data into a raster
  img <- as.raster(y)
  
  if (length(axes) == 1) axes <- rep(axes,2)
  # initialise the plot area, axes and labels appropriately
  plot.new(); plot.window(xlim, ylim, xaxs="i", yaxs="i"); box()
  for(i in seq_along(axes))
    axis(i, labels=axes[[i]], lwd=0, lwd.ticks=1)

  blocks <- c(0,which(diff(which) != 1),length(which))
  for (i in 1:(length(blocks)-1)) {
    ii <- which[seq(blocks[[i]]+1,blocks[[i+1]])]
    rasterImage(img[ii,],0.5,max(ii)+0.5,ncol(y)+0.5,min(ii)-0.5,interpolate=FALSE)
  }
}

# Read the CSV format MCMC output from MDI
readMdiMcmcOutput <- function (path, as.is=TRUE, check.names=FALSE, ...) {
  # load the MCMC output
  mcmc <- read.csv(path, as.is=as.is, check.names=check.names, ...)
  
  # make sure it looks like the sort of MDI output we support
  nfiles <- length(grep('^MassParameter_[1-9][0-9]*$',
                        colnames(mcmc)))
  # can determine everything else from the number of files and number of columns at the moment
  npar  <- nfiles * (1+(nfiles-1)/2)
  ncols <- round((ncol(mcmc) - npar) / nfiles)
  if (npar + ncols * nfiles != ncol(mcmc))
    stop("Invalid MDI file format")
  # generate a series of substrings
  pats <- sprintf('MassParameter_%i', 1:nfiles)
  if (nfiles > 1) {
    for (i in 1:(nfiles-1)) {
      pats <- c(pats,sprintf("Phi_%i%i",i,(i+1):nfiles))
    }
  }
  pats <- c(pats, sprintf('Dataset%i_', rep(1:nfiles,each=ncols)))
  # ensure the column names match
  stopifnot(pats == substr(colnames(mcmc), 1, nchar(pats)))
  # columns are OK, but buffering can cause partial lines to be written to the
  # CSV file, drop the final line if it looks like this has happened
  if(any(is.na(mcmc[nrow(mcmc),])))
    mcmc <- mcmc[-nrow(mcmc),]
  
  class(mcmc) <- c("MdiMcmcOutput",class(mcmc))
  attributes(mcmc) <- within(attributes(mcmc), {
    nfiles <- nfiles
    nitems <- ncols
    nphi   <- nfiles*(nfiles-1)/2
    allocs <- lapply(1:nfiles, function(k) npar+nitems*(k-1)+(1:nitems))
  })
  # finally, return the data back to the user
  mcmc
}

# support subsetting specially, namely copy our attributes over
"[.MdiMcmcOutput" <- function(mcmc,i,j) {
  # if we subset on columns we just end up with a "normal" dataset
  if(!missing(j)) {
    return(NextMethod(i,j))
  }
  # otherwise, maintain special handling
  out <- NextMethod(i,)
  class(out) <- class(mcmc)
  attributes(out) <- append(attributes(out),
                            attributes(mcmc)[c("nfiles","nitems","nphi","allocs")])
  out
}

# Thin a MC chain.  A maximum of @n evenly spaced samples are returned,
# starting from @from*100 percent of the way through the chain.
tail.MdiMcmcOutput <- function (mcmc, n=100, from=0.5) {
  m <- nrow(mcmc)
  # turn the @from ratio into a sample number
  from <- max(1,1+floor(from*(m-1)))
  ii <- seq.int(from,m,length.out=min(m-from+1,n))
  mcmc[unique(round(ii)),]
}

# extract the latent allocation variables from MC output @m for dataset @k
getMdiAllocations <- function (mcmc, k) {
  allocs <- attr(mcmc, "allocs")
  if (k < 1 || k > length(allocs))
    stop("the requested dataset doesn't exist in this chain")
  ii <- allocs[[k]]
  alloc <- as.matrix(mcmc[,ii])
  colnames(alloc) <- sub(sprintf('^Dataset%i_', k),'',colnames(mcmc)[ii])
  alloc
}

getClustersOccupied <- function (mcmc) {
  sapply(1:attr(mcmc,"nfiles"), function(k) {
    apply(getMdiAllocations(mcmc, k), 1, function(x) length(unique(x)))
  })
}

# turn the allocations into a PSM
#  NOTE: you will almost certainly want to thin before doing this!
genPosteriourSimilarityMatrix <- cmpfun(function (m) {
  n <- ncol(m)
  o <- matrix(0,n,n,dimnames=list(colnames(m),colnames(m)))
  for (i in 1:nrow(m)) {
    a <- as.numeric(m[i,])
    for (j in 1:n) {
      o[j,] <- o[j,] + (a[[j]] == a)
    }
  }
  1-o/nrow(m)
})

plotSingleMcmcSample <- function(mcmc, datafiles) {
  # check the caller has selected a single MCMC sample for us to look at
  stopifnot(nrow(mcmc) == 1)
  
  nfiles <- attr(mcmc,"nfiles")
  
  # make sure it's consistent with the other parameters
  stopifnot(nfiles == length(datafiles))
  
  # we want all unique cluster allocations over every dataset, so just strip out DP masses and phis
  clus <- sort(unique(as.numeric(mcmc[-(1:(nfiles*(nfiles+1)/2))])))
  
  # configure the plot area for what we're about to do
  par0 <- par(mfrow=c(nfiles,length(clus)),
              mar=c(1,0.5,0.5,0.5),
              oma=c(1,3,0.5,0),
              mgp=c(3,0.6,0))
  # run down each file
  for (k in 1:nfiles) {
    # get data for this file
    xx <- datafiles[[k]]
    # get allocations and figure out how this file records time
    alloc <- as.numeric(getMdiAllocations(mcmc,k))
    
    # loop through every cluster
    n = 1
    for (i in clus) {
      # plot the data for this cluster
      plot(xx, which=which(alloc==i), axes=c(TRUE, n == 1))
      # name this row/dataset
      if (n == 1)
        mtext(names(datafiles)[[k]], side=2, line=2)
      n <- n + 1
    }
  }
  # restore the plot area
  par(par0)
}

# plot allocation agreement between all datasets appearing in @mcmc
plotAllocationAgreement <- function (mcmc, palfn=rainbow) {
  nfiles <- attr(mcmc,"nfiles")
  nitems <- attr(mcmc,"nitems")

  if (nfiles < 2)
    stop("Can't check agreement between all pairwise combinations of one dataset")
  
  allocs <- lapply(1:nfiles, function(i) getMdiAllocations(mcmc,i))
  
  time <- as.numeric(rownames(mcmc))
  col <- palfn(nitems)
  # set plot parameters
  par0 <- par(mfrow=rep(nfiles-1,2),
              mar=c(1,0.5,0,0),
              oma=c(2.5,2.5,0.5,0.5),
              mgp=c(3,0.6,0))
  # force a redraw
  plot.new()
  for (i in 1:(nfiles-1)) {
    # pull out allocations for file @i as they will be used a few times
    ai <- getMdiAllocations(mcmc,i)
    for (j in (i+1):nfiles) {
      # move to the correct place
      par(mfg=c(j-1,i))
      allocagree <- apply(allocs[[i]] == allocs[[j]], 2, function(x) cumsum(x) / seq(1,length(x)))
      plot.new(); plot.window(range(time),0:1); box(bty="l")
      if (j == nfiles) mtext(attr(datafiles[[i]],"name"), side=1, line=2)
      if (i == 1)      mtext(attr(datafiles[[j]],"name"), side=2, line=2)
      axis(1, labels=j==nfiles,lwd=0,lwd.ticks=1)
      axis(2, labels=i==1,lwd=0,lwd.ticks=1)
      for (k in 1:ncol(allocagree)) lines(time, allocagree[,k], col=col[[k]])
    }
  }
  par(par0)
}

# plot all pairwise allocation agreements between files as a series of masses
plotAllocationAgreementHistogram <- function(mcmc, datafiles) {
  nfiles <- attr(mcmc,"nfiles")
  nitems <- attr(mcmc,"nitems")
  
  # set plot parameters
  par0 <- par(mfrow=rep(nfiles-1,2),
              mar=c(1,0.5,0,0),
              oma=c(2.5,1.5,0.5,0.5),
              mgp=c(3,0.6,0))
  # force a redraw
  plot.new()
  for (i in 1:(nfiles-1)) {
    # pull out allocations for file @i as they will be used a few times
    ai <- getMdiAllocations(mcmc,i)
    for (j in (i+1):nfiles) {
      # move to the correct place
      par(mfg=c(j-1,i))
      # configure a new plot
      plot.new(); plot.window(c(0,ncol(ai)),c(0,1)); box(bty="l")
      axis(1,labels=j==nfiles,lwd=0,lwd.ticks=1)
      if (i == 1)      mtext(attr(datafiles[[j]],"name"), side=2, line=0.5)
      if (j == nfiles) mtext(attr(datafiles[[i]],"name"), side=1, line=2)
      # get number of matching entries between file @i and @j
      matching <- apply(ai == getMdiAllocations(mcmc,j),1,sum)
      # dotted line for mean matches between files
      abline(v=mean(matching), lwd=0.5,lty=3)
      # histogram for full posteriour summary
      lines(table(matching)/length(matching))
    }
  }
}

# generate a consensus clustering for the MCMC chain stored in m, that
# used "nfiles" data files, with "items" in each file
generateConsensusPSM <- function (mcmc) {
  nfiles <- attr(mcmc,"nfiles")
  nitems <- attr(mcmc,"nitems")
    
  cc <- list()
  ccs <- matrix(0,nitems,nitems)
  for (i in 1:nfiles) {
    cc[[i]] <- genPosteriourSimilarityMatrix(getMdiAllocations(mcmc,i))
    ccs <- ccs + cc[[i]]
  }
  ccs <- ccs/nfiles
  
  hc <- hclust(as.dist(ccs),method="average")
  
  list(nfiles=nfiles,nitems=nitems,cc=cc,ccs=ccs,hc=hc)
}

# plot the consensus PSMs
# one big one for the consensus and some smaller ones for the individual datasets
plotConsensusPSM <- function (cpsm, datafiles, cut, pal=gray.colors(64), col.box="royalblue", ann=TRUE, ann.cex=0.2) {
  nfiles <- cpsm$nfiles
  items  <- cpsm$nitems
  ord    <- cpsm$hc$ord
  clus   <- rle(cutree(cpsm$hc,cut)[ord])$len
  clusf  <- cumsum(clus)
  clusr  <- cumsum(rev(clus))
  names  <- lapply(datafiles, function(d) attr(d,"name"))

  drawme <- function (cc, ann) {
    cc <- cc[rev(ord),ord]
    image(1:items,1:items, cc, zlim=0:1, col=pal, useRaster=TRUE, axes=FALSE, ann=FALSE)
    for (i in c(1,3)) {
      axis(i,   c(0,clusr)+0.5, FALSE, lwd=0, lwd.ticks=1)
      axis(i+1, c(0,clusf)+0.5, FALSE, lwd=0, lwd.ticks=1)
    }
    box(col=col.box)
    if(ann) {
      mtext(colnames(cc),2,at=1:items,cex=ann.cex,las=2, line=0.2)
    }
  }
  
  if (nfiles > 1) {
    lo <- matrix(0,nfiles,2)
    lo[1:nfiles,1] <- 1
    lo[1:nfiles,2] <- 1+(1:nfiles)
    layout(lo, c(nfiles,1))
  }
  drawme(cpsm$ccs, ann)
  if (nfiles > 1) {
    for (i in 1:nfiles) {
      drawme(cpsm$cc[[i]], ann=FALSE)
      mtext(names[[i]],2,1)
    }
  }
}

extractPSMClustPartition <- function (cpsm, cuts, datafiles=NULL) {
    xx <- sapply(1:length(cpsm$cc), function(i)
                 cutree(cpsm$hc,cuts[[i]]))
    if(!is.null(datafiles))
        colnames(xx) <- lapply(datafiles, function(x) attr(x,"name"))
    xx
}
