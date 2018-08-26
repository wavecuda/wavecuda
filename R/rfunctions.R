#' check.trans.inputs
#' Checks the inputs for a transformation and converts them into the required format for the library.
#' Internal function
#'
#' @param xin Input vector, normally a vector to be transformed into the wavelet domain
#' @param direction "FWD" or "BWD"
#' @param nlevels Number of levels of the wavelet transform to perform
#' @param transform.type "DWT" or "MODWT" (time ordered)
#' @param filter Wavelet filter to use
#'
#' @export
check.trans.inputs <- function(xin,direction,nlevels,transform.type,filter){
    ttype <- switch(transform.type,"DWT"=0,"MODWT"=1)
    if(is.null(ttype)) stop("Unexpected transform type")
    
    filt <- switch(filter,"Haar"=1,"D4"=2,"C6"=3,"LA8"=4)
    if(is.null(filt)) stop("Unexpected filter")

    filtlen <- switch(filter,"Haar"=2,"D4"=4,"C6"=6,"LA8"=8)
    
    sense <- switch(direction,"FWD"=1,"BWD"=0)
    if(is.null(sense)) stop("Unexpected direction!")

    lenxin <- length(xin) ## could be len or len*2*nlevels

    if(!is.numeric(xin)) stop("Unexpected input vector")

    if(ttype == 0){
        ## DWT. We only use xin, which we set to x
        x <- as.double(xin)
        xmod <- numeric(0)
        len <- lenxin
        if(log2(len) != round(log2(len))) stop("Need length to be power of 2")
    }else{
        ## MODWT. We have two options...
        if(sense==1){
            ## FWD: xin is our input, x
            ## and we need to allocate xmod
            x <- as.double(xin)
            len <- lenxin
            if(log2(len) != round(log2(len))) stop("Need length to be power of 2")
            xmod <- rep(0,len*2*log2(len))
        }else{
            ## BWD: xin is xmod, a transformed MODWT vector
            ## and we need to allocate x in which to store
            ## the reconstructed vector
            xmod <- as.double(xin)
            len <- lenxin/(2*nlevels)
            if(log2(len) != round(log2(len))) stop("Was this really a MODWT vector? If so, check nlevels. I calculate the reconstruction vector to be of inappropriate size.")
            x <- as.double(rep(0,len))
        }    
    }
    maxlevels <- floor(log2(len/filtlen))+1
    if(nlevels == 0){
        nlevels <- maxlevels
        if( (direction=="BWD") & (ttype>0) ) stop("Probably got an error of infinite length: len = ",len)
    }
    if( (nlevels != round(nlevels)) | (nlevels<0) | (nlevels>maxlevels) ){
        stop("nlevels should be an integer [whole number] between 1 and ",log2(len))
    }

    list(x=x,xmod=xmod,len=len,sense=sense,nlevels=nlevels,ttype=ttype,filt=filt,filtlen=filtlen)
}

#' check.thresh.inputs
#' Checks the inputs for thresholding and converts them into the required format for the library
#' Internal function
#'
#' @param xwav WST object 
#' @param hard.soft "hard" or "soft" thresholding
#' @param thresh Threshold value
#' @param min.level Minimum level for thresholding
#' @param max.level Maximum level for thresholding
#'
#' @export
check.thresh.inputs <- function(xin,nlevels,transform.type, filter,hard.soft,thresh,min.level,max.level){
    arg.list <- check.trans.inputs(xin,"BWD",nlevels,transform.type,filter)
    ## we're not doing a BWD transform, but just getting the inputs right!

    arg.list$sense <- NULL
    ## just removing this list element as not needed
    
    if( (!is.atomic(thresh)) | (length(thresh)>1) ) stop("Only scalar thresholds supported")
    if(thresh<=0) stop("Threshold must be greater than 0")

    arg.list$thresh <- thresh

    arg.list$hardness <- switch(hard.soft,"hard"=0,"soft"=1)
    if(is.null(arg.list$hardness)) stop("Only hard or soft thresholding supported")
    
    if( (min.level!= round(min.level)) | (max.level!= round(max.level)) |
        (min.level > max.level) | (min.level <= 0) | (max.level > nlevels) ){
        stop("min.level and max.level must be integers in the range 1...nlevels")
    }
    
    arg.list$min.level <- min.level - 1
    arg.list$max.level <- max.level - 1
    ## our levels in C is C-style, i.e. 0 -> J-1
    ## we change it in R to be like R indexing, i.e. 1 -> J

    return(arg.list)
}

#' check.smooth.inputs
#' Checks the inputs for smoothing and converts them into the required format for the library
#' Internal function
#'
#' @param xin Input vector, normally a vector to be transformed into the wavelet domain
#' @param nlevels Number of levels of the wavelet transform to perform
#' @param transform.type "DWT" or "MODWT" (time ordered)
#' @param filter Wavelet filter to use
#' @param thresh.type "manual", "univ" or "cv"
#' @param hard.soft "hard" or "soft" thresholding
#' @param thresh Threshold value
#' @param min.level Minimum level for thresholding
#' @param max.level Maximum level for thresholding
#' @param tol Tolerance for cross validation smoothing
#'
#' @export
check.smooth.inputs <- function(xin,nlevels,transform.type,filter,thresh.type,thresh,hard.soft,min.level,max.level,tol){
    arg.list <- check.trans.inputs(xin,"FWD",nlevels,transform.type,filter)

    arg.list$sense <- NULL
    ## just removing this list element as not needed

    arg.list$threshtype <-  switch(thresh.type,"manual"=0,"univ"=1,"cv"=2)
    if(is.null(arg.list$threshtype)) stop("Only manual, univ or cv supported for threshold type")

    if(thresh.type=="manual"){
        if( (!is.atomic(thresh)) | (length(thresh)>1) ) stop("Only scalar thresholds supported")
        if(thresh<=0) stop("Threshold must be greater than 0")        
        arg.list$thresh <- thresh
    }else{
        arg.list$thresh <- numeric(0)
    }

    if(thresh.type=="cv"){
        if(tol<=0) stop("Tolerance must be greater than 0")        
        arg.list$tol <- tol
    }else{
        arg.list$tol <- numeric(0)
    }
    
    arg.list$hardness <- switch(hard.soft,"hard"=0,"soft"=1)
    if(is.null(arg.list$hardness)) stop("Only hard or soft thresholding supported")
    
    if( (min.level!= round(min.level)) | (max.level!= round(max.level)) |
        (min.level > max.level) | (min.level <= 0) | (max.level > nlevels) ){
        stop("min.level and max.level must be integers in the range 1...nlevels")
    }
    
    arg.list$min.level <- min.level - 1
    arg.list$max.level <- max.level - 1
    ## our levels in C is C-style, i.e. 0 -> J-1
    ## we change it in R to be like R indexing, i.e. 1 -> J

    return(arg.list)
    
}

#' return.trans
#' Puts the results into a nice structure for returning the result of a transform.
#' Internal function
#'
#' @param arglist Input arguments for the transform
#' @param argsin Input arguments for the transform
#'
#' @export
return.trans <- function(arglist, argsin){
    ## returning a nice wavelet structure

    ## modify to return transform vector always in same value
    
    wvt_return <- structure(list("x" = arglist$x,
                                 "ttype" = argsin$ttype,
                                 "filt" = argsin$filter,
                                 "filtlen" = arglist$filterlen,
                                 "nlevels" = arglist$nlevels,
                                 "len" = arglist$len,
                                 "xmod" = arglist$xmod),
                            class = "WST")
    
    return(wvt_return)
}

#' return.thresh
#' Puts the results into a nice structure for returning the result of thresholding.
#'
#' @param arglist Input arguments for the transform
#'
#' @export
return.thresh <- function(arglist){
    ## to
    if(arglist$ttype>0){
        return(arglist$xmod)
    }
    else{
        return(arglist$x)
    }
}

#' CPUTransform
#'
#' Wavelet transform using the CPU.
#' The input vector can be in the time domain or wavelet domain.
#' Supports DWT and (time-ordered) MODWT.
#' Supports Haar, D4, C6 and LA8 filters.
#' Allows the user to specify number of levels of transform required. Note that the maximum number of levels implemented is \eqn{\log_2{n} - b+1} where \eqn{n} is the length of the input vector and \eqn{b = ceiling (\log_2{L})} where \eqn{L} is the filter length. This means that we only allow transformations up to the level where the filter does not wrap around the coefficients more than once: with the Haar filter we do the full dyadic transform, whereas with the LA8 filter we stop after filtering 8 coefficients.
#'
#' The DWT is transformed in-place after copying the input vector, whereas the MODWT requires extra memory allocation. The structure of the transformed DWT vector is the standard interleaved form. The structure of the transformed MODWT vector is \eqn{n} scaling coefficients then detail coefficients, concatenated sequentially for each layer. The best way to access the coefficients for each level is via \code{\link{WST.to.DT}}.
#' 
#' @param xin Vector input
#' @param direction "FWD" or "BWD"
#' @param nlevels Number of levels of transform; 0 means full transform
#' @param transform.type "DWT" or "MODWT"
#' @param filter e.g. "Haar"
#'
#' @return Returns a WST object containing the transform and details.
#'
#' @seealso \code{\link{GPUTransform}}, \code{\link{CPUThreshold}}, \code{\link{CPUSmooth}}, \code{\link{GPUSmooth}}, \code{\link{WST.to.DT}}
#' 
#' @useDynLib wavecuda RcpuTransform
#' @export
CPUTransform <- function(xin, direction, nlevels, transform.type, filter){

    ## add WST object creation...

    args.in <- list(ttype=transform.type,
                    filter=filter)
    
    arg.list <- check.trans.inputs(xin, direction, nlevels, transform.type, filter)
    
    arg.list <- .C("RcpuTransform",
                   x=arg.list$x,
                   xmod=arg.list$xmod,
                   len=as.integer(arg.list$len),
                   sense=as.integer(arg.list$sense),
                   nlevels=as.integer(arg.list$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   PACKAGE="wavecuda")
    
    return(return.trans(arg.list, args.in))
}

#' @useDynLib wavecuda RgpuTransform
#' @export
GPUTransform <- function(xin, direction, nlevels, transform.type, filter){

    ## add WST object creation...
    args.in <- list(ttype=transform.type,
                    filter=filter)
    
    arg.list <- check.trans.inputs(xin, direction, nlevels, transform.type, filter)
    
    arg.list <- .C("RgpuTransform",
                   x=arg.list$x,
                   xmod=arg.list$xmod,
                   len=as.integer(arg.list$len),
                   sense=as.integer(arg.list$sense),
                   nlevels=as.integer(arg.list$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   PACKAGE="wavecuda")
    
    return(return.trans(arg.list, args.in))
}

## #' @useDynLib wavecuda RcpuThreshold
## #' @export
## CPUThreshold <- function(xin,nlevels,transform.type,filter,thresh,hard.soft,min.level,max.level){
##     arg.list <- check.thresh.inputs(xin,nlevels,transform.type, filter,hard.soft,thresh,min.level,max.level)

##     arg.list <- .C("RcpuThreshold",
##                    x=arg.list$x,
##                    xmod=arg.list$xmod,
##                    len=as.integer(arg.list$len),
##                    nlevels=as.integer(arg.list$nlevels),
##                    ttype=as.integer(arg.list$ttype),
##                    filter=as.integer(arg.list$filt),
##                    filterlen=as.integer(arg.list$filtlen),
##                    thresh=arg.list$thresh,
##                    hardness=as.integer(arg.list$hardness),
##                    minlevel=as.integer(arg.list$min.level),
##                    maxlevel=as.integer(arg.list$max.level),
##                    PACKAGE="wavecuda")

##     return(return.thresh(arg.list))
## }

#' @useDynLib wavecuda RcpuThreshold
#' @export
CPUThreshold <- function(xwav,thresh,hard.soft,min.level,max.level){
    arg.list <- check.thresh.inputs(xin,nlevels,transform.type, filter,hard.soft,thresh,min.level,max.level)

    arg.list <- .C("RcpuThreshold",
                   x=arg.list$x,
                   xmod=arg.list$xmod,
                   len=as.integer(arg.list$len),
                   nlevels=as.integer(arg.list$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   thresh=arg.list$thresh,
                   hardness=as.integer(arg.list$hardness),
                   minlevel=as.integer(arg.list$min.level),
                   maxlevel=as.integer(arg.list$max.level),
                   PACKAGE="wavecuda")

    return(return.thresh(arg.list))
}


#' @useDynLib wavecuda RcpuSmooth
#' @export
CPUSmooth <- function(xin,nlevels,transform.type,filter,thresh.type,thresh=NULL,hard.soft,min.level,max.level,tol=0.01){

    arg.list <- check.smooth.inputs(xin,nlevels,transform.type,filter,thresh.type,thresh,hard.soft,min.level,max.level,tol)
    
    arg.list <- .C("RcpuSmooth",
                   x=arg.list$x,
                   len=as.integer(arg.list$len),
                   nlevels=as.integer(arg.list$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   threshtype=as.integer(arg.list$threshtype),
                   thresh=arg.list$thresh,
                   hardness=as.integer(arg.list$hardness),
                   minlevel=as.integer(arg.list$min.level),
                   maxlevel=as.integer(arg.list$max.level),
                   tol=arg.list$tol,
                   PACKAGE="wavecuda")

    return(arg.list$x)
}

#' @useDynLib wavecuda RgpuSmooth
#' @export
GPUSmooth <- function(xin,nlevels,transform.type,filter,thresh.type,thresh=NULL,hard.soft,min.level,max.level,tol=0.01){

    arg.list <- check.smooth.inputs(xin,nlevels,transform.type,filter,thresh.type,thresh,hard.soft,min.level,max.level,tol)

    if(thresh.type=="univ") stop("Universal threshold not [yet] implemented on GPU, as it's probably quicker on CPU")
    
    arg.list <- .C("RgpuSmooth",
                   x=arg.list$x,
                   len=as.integer(arg.list$len),
                   nlevels=as.integer(arg.list$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   threshtype=as.integer(arg.list$threshtype),
                   thresh=arg.list$thresh,
                   hardness=as.integer(arg.list$hardness),
                   minlevel=as.integer(arg.list$min.level),
                   maxlevel=as.integer(arg.list$max.level),
                   tol=arg.list$tol,
                   PACKAGE="wavecuda")

    return(arg.list$x)
}

## #' @useDynLib wavecuda RgpuTransformList
## #' @export
GPUTransformList <- function(xin, direction, nlevels, transform.type, filter){
    ## xin should be a list
    ## direction, nlevels, transform.type, filter should be vectors
    ## all of the same length...
    
    len <- length(xin)

    if(len<=1) stop("We require a list of xin vectors of length > 1")

    if( (length(direction) != len) || (length(nlevels) != len) || (length(transform.type) != len) || (length(filter) != len) ) stop("Inconsistent length of input list & vectors")

    arg.list <- list()

    for(il in 1:len){
        arg.list[[il]] <- check.trans.inputs(xin[[il]], direction[il], nlevels[il], transform.type[il], filter[il])
    }

    ret.list <- .Call("RgpuTransformList",
                      arglist <- arg.list,
                      package="wavecuda")

    print("Done....")

    return(ret.list)
    
}

## #' @export
wstCV1 <- function (ndata, ll = 3, type = "soft", filter.number = 10, family = "DaubLeAsymm", 
    tol = 0.01, verbose = 0, plot.it = FALSE, norm = l2norm, 
    InverseType = "average", uvdev = madmad) 
{
    ## edit of Nason's wstCV for diagnostics
    nlev <- log(length(ndata))/log(2)
    levels <- ll:(nlev - 1)
    nwst <- wst(ndata, filter.number = filter.number, family = family)
    uv <- threshold(nwst, levels = levels, type = type, policy = "universal", 
        dev = madmad, return.thresh = TRUE)[1]
    if (verbose == 1) 
        cat("Now optimising cross-validated error estimate\n")
    levels <- ll:(nlev - 2)
    R <- 0.61803399
    C <- 1 - R
    ax <- 0
    bx <- uv/2
    cx <- uv
    x0 <- ax
    x3 <- cx
    if (abs(cx - bx) > abs(bx - ax)) {
        x1 <- bx
        x2 <- bx + C * (cx - bx)
    }
    else {
        x2 <- bx
        x1 <- bx - C * (bx - ax)
    }
    fa <- GetRSSWST(ndata, threshold = ax, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    ## cat("Done 1\n")
    fb <- GetRSSWST(ndata, threshold = bx, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    ## cat("Done 2\n")
    fc <- GetRSSWST(ndata, threshold = cx, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    ## cat("Done 3\n")
    f1 <- GetRSSWST(ndata, threshold = x1, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    ## cat("Done 4\n")
    f2 <- GetRSSWST(ndata, threshold = x2, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    ## cat("Done 5\n")
    xkeep <- c(ax, cx, x1, x2)
    fkeep <- c(fa, fc, f1, f2)
    if (plot.it == TRUE) {
        plot(c(ax, bx, cx), c(fa, fb, fc))
        text(c(x1, x2), c(f1, f2), lab = c("1", "2"))
    }
    cnt <- 3
    while (abs(x3 - x0) > tol * (abs(x1) + abs(x2))) {
        if (verbose > 0) {
            cat("x0=", x0, "x1=", x1, "x2=", x2, "x3=", x3, "\n")
            cat("f1=", f1, "f2=", f2, "\n")
        }
        if (f2 < f1) {
            x0 <- x1
            x1 <- x2
            x2 <- R * x1 + C * x3
            f1 <- f2
            f2 <- GetRSSWST(ndata, threshold = x2, levels = levels, 
                type = type, filter.number = filter.number, family = family, 
                norm = norm, verbose = verbose, InverseType = InverseType)
            if (verbose == 2) {
                cat("SSQ: ", signif(f2, digits = 3), "\n")
            }
            else if (verbose == 1) 
                cat(".")
            xkeep <- c(xkeep, x2)
            fkeep <- c(fkeep, f2)
            if (plot.it == TRUE) 
                text(x2, f2, lab = as.character(cnt))
            cnt <- cnt + 1
        }
        else {
            x3 <- x2
            x2 <- x1
            x1 <- R * x2 + C * x0
            f2 <- f1
            f1 <- GetRSSWST(ndata, threshold = x1, levels = levels, 
                type = type, filter.number = filter.number, family = family, 
                norm = norm, verbose = verbose, InverseType = InverseType)
            if (verbose == 2) 
                cat("SSQ: ", signif(f1, digits = 3), "\n")
            else if (verbose == 1) 
                cat(".")
            xkeep <- c(xkeep, x1)
            fkeep <- c(fkeep, f1)
            if (plot.it == TRUE) 
                text(x1, f1, lab = as.character(cnt))
            cnt <- cnt + 1
        }
    }
    if (f1 < f2) 
        tmp <- x1
    else tmp <- x2
    x1 <- tmp/sqrt(1 - log(2)/log(length(ndata)))
    if (verbose == 1) 
        cat("Correcting to ", x1, "\n")
    else if (verbose == 1) 
        cat("\n")
    g <- sort.list(xkeep)
    xkeep <- xkeep[g]
    fkeep <- fkeep[g]
    if (verbose >= 1) {
        cat("Reconstructing CV \n")
    }
    nwstT <- threshold(nwst, type = type, levels = levels, policy = "manual", 
        value = x1)
    nwstT <- threshold(nwstT, type = type, levels = nlevelsWT(nwstT) - 
        1, policy = "universal", dev = uvdev)
    ## not sure why he does an extra universal threshold here!
    xvwr <- AvBasis.wst(nwstT)
    list(ndata = ndata, xvwr = xvwr, xvwrWSTt = nwstT, uvt = uv, 
        xvthresh = x1, xkeep = xkeep, fkeep = fkeep)
}

#'@export
print.WST <- function(x){
    cat("--------------------------------------\n")
    cat("Wavecuda STructure object WST class")
    cat("\n of type:                  ",x$ttype)
    cat("\n with filter:              ",x$filt)
    cat("\n levels of transform:      ",x$nlevels)
    cat("\n original vector of length:",x$len)
    cat("\n")
    cat("--------------------------------------\n")
}

#' @import data.table
#' @export
WSTtoDT <- function(Xwav, scaling = TRUE, forPlotting = FALSE){
    levelList <- sapply(X = 1:Xwav$nlevels, FUN = function(l) getCoeffLevel(Xwav,l,"d"), simplify = F)

    if(forPlotting)
        scaling <- FALSE
    
    ## we make placeholders for min/max treatment
    ## because we can't plot y_free with symmetric axes
    ## so we create dummy rows at the end of the data frame
    ## all with translate value of 0
    ## and initialise these with NA for W
    padNA <- rep(NA, Xwav$nlevels * 2)
    pad0 <- rep(0, Xwav$nlevels * 2)
    padl <- rep(1:Xwav$nlevels, each = 2) # levels
    padmm <- rep(1:2, Xwav$nlevels) # minmax

    if(Xwav$ttype == "DWT"){
        xw_df <- data.table(W = c(Xwav$x, padNA),
                            Level = c(makeLevelNVec(Xwav,levelList), padl),
                            Translate = c(makeLevelTVec(Xwav,levelList), pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$x)),padmm)
                            ## 0 for values of the transform
                            ## 1 for  min(-abs(W))
                            ## 2 for  max(abs(W)) per level
                            )
        ## xw_df[,T2 := ifelse(.I <= Wxwav$len, (.I-1)/2,0)]
        ## xw_df[,L2 := quickLVec(.I)]
        ## not correct
    }
    if(Xwav$ttype == "MODWT"){
        detailCoeffSelector <- (1:Xwav$len) + Xwav$len + rep(2*(0:(Xwav$nlevels-1))*Xwav$len,each = Xwav$len)
        xw_df <- data.table(W = c(Xwav$xmod[detailCoeffSelector], padNA),
                            Level = c(makeLevelNVec(Xwav,levelList),padl),
                            Translate = c(makeLevelTVec(Xwav, levelList),pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$xmod)/2),padmm)
                            ## 0 for values of the transform
                            ## 1 for  min(-abs(W))
                            ## 2 for  max(abs(W)) per level
                            )
    }
    xw_df[, W := replace(W, Translate == 0, max(abs(W), na.rm = T)),
          by = c("Level")]
    ## replace the created NA values with max abs per level
    xw_df[(Translate == 0) & (minmax == 1), W:=-W]
    ## set the min vals per level
    xw_df <- xw_df[Level >0]
    ## filter -> will be done below

    if(!forPlotting){
        xw_df <- xw_df[minmax == 0]
        ## remove min max extra values
        xw_df[,minmax := NULL]
        ## remove min max column
    }
    
    return(xw_df)
}

#' @import data.table
#' @import wavethresh
#' @export
WSTtowavethresh <- function(XW, showWarnings = TRUE){
    xw_df <- WST.to.DT(XW)
    ## need scaling too :)

    filter_number = switch(XW$filt,"Haar"=1,"D4"=2,"C6"=3, "LA"=4)
    family = switch(XW$filt,"Haar"="DaubExPhase","D4"="DaubExPhase","C6"="DaubLeAsymm", "LA"="DaubLeAsymm")
    
    if((XW$filt == "C6") & showWarnings)
        warning("Coiflets not implemented in wavethresh; you will not be able reconstruct with wavethresh")

    if((XW$ttype == "MODWT") & showWarnings)
        warning("Wavethresh uses a different ordering in MODWT coefficients")
    
    XW_wavethresh <- wavethresh::wd(rep(0,XW$len),
                        filter.number = filter_number,
                        family = family,
                        type = switch(XW$ttype, "DWT" = "wavelet", "MODWT" = "station"),
                        bc = "periodic")
    
    for(l in 1:XW$nlevels){
        XW_wavethresh <- putD(XW_wavethresh,
                              level = XW_wavethresh$nlevels - l,
                              v = xw_df[Level == l, W])
    }

    warning("Haven't yet implemented scaling coeffs transfer")
    ## need to add scaling coeffs for top layers
    
    return(XW_wavethresh)
}
