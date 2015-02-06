# random draw of @n items from dirichlet process with concentration @alpha
rdirichletprocess <- function (n, alpha=1) {
  a <- 1
  for (i in 2:n) {
    a[i] <- sample(1:(max(a)+1), 1, prob=c(table(a),alpha))
  }
  a
}

create.normgam <- function (clus, nfeat) {
  nclus <- max(clus)
  means <- matrix(rnorm(nclus*nfeat,0,3), nclus)
  matrix(rnorm(length(clus)*nfeat,means[clus,]),
         length(clus),
         dimnames=list(sprintf("c%i",clus),sprintf("f%i",1:nfeat)))
}

create.multinom <- function (clus, levels) {
  out <- matrix(0,length(clus),length(levels),
                dimnames=list(sprintf("c%i",clus),sprintf("f%i",seq_along(levels))))
  nclus <- max(clus)
  for (i in seq_along(levels)) {
    l <- levels[[i]]
    pp <- matrix(rgamma(nclus * l, 0.2), nclus)
    for (j in 1:nclus) {
      jj <- clus == j
      out[jj,i] <- sample.int(l, sum(jj), replace=TRUE, prob=pp[j,])
    }
  }
  out
}

create.bagofwords <- function (clus, cnt, nfeat) {
  out <- matrix(0,length(clus),nfeat,
                dimnames=list(sprintf("c%i",clus),
                              sprintf("f%i",1:nfeat)))
  nclus <- max(clus)
  for (j in 1:nclus) {
    jj <- clus == j
    out[jj,] <- t(rmultinom(sum(jj), cnt, rgamma(nfeat, 1)))
  }
  out
}

# squared exponential kernel
sqexp <- function (t, l2, s2)
  s2 * exp(-(matrix(t, length(t), length(t), byrow=TRUE) - t)^2 / (2*l2))

create.gaussianprocess <- function (alloc, nfeat) {
  # generate time series - shared between all groups
  time <- round(cumsum(rgamma(nfeat,2,5)),3)
  
  out <- matrix(NA,length(alloc),nfeat,dimnames=list(sprintf("c%i",alloc),
                                              time))
  
  for (i in unique(alloc)) {
    len2 <- 1/rgamma(1,3,3)
    sig2 <- 1/rgamma(1,3,3)
    noi2 <- 1/rgamma(1,2,1)
    
    mean <- MASS::mvrnorm(1, rep(0,length(time)), sqexp(time, len2, sig2))
    ii <- which(alloc == i)
    out[ii,] <- matrix(rnorm(nfeat * length(ii),mean,noi2),ncol=nfeat,byrow=TRUE)
  }
  out
}
