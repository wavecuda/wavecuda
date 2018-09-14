#' ttype.convertor
#' Converts transform types between the format required for R/CUDA libraries.
#' Internal function
#'
#' @param transform.type "DWT" or "MODWT"
#'
ttype.convertor <- function(transform.type, toCUDA = TRUE){
  if(toCUDA){
    return(switch(transform.type,"DWT"=0,"MODWT"=1))
  }else{
    return(ifelse(transform.type==0,"DWT","MODWT"))
  }
}

#' filt.convertor
#' Converts filter identifiers between the format required for R/CUDA libraries.
#' Internal function
#'
#' @param filter Filter name
#'
filt.convertor <- function(filter, toCUDA = TRUE){
  if(toCUDA){
    return(switch(filter,"Haar"=1,"D4"=2,"C6"=3,"LA8"=4))
  }else{
    return(switch(filter,"Haar","D4","C6","LA8"))
  }
}

#' filtlen.calculator
#' Returns the filter length for a given filter name.
#' Internal function
#'
#' @param filter Filter name
#'
filtlen.calculator <- function(filter){
  return(switch(filter,"Haar"=2,"D4"=4,"C6"=6,"LA8"=8))
}

#' sense.convertor
#' Converts direction identifiers between the format required for R/CUDA libraries.
#' Internal function
#'
#' @param direction Transform direction
#'
sense.convertor <- function(direction, toCUDA = TRUE){
  if(toCUDA){
    return(switch(direction,"FWD"=1,"BWD"=0))
  }else{
    return(ifelse(direction=="FWD",1,0))
  }
}

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
    ttype <- ttype.convertor(transform.type)
    if(is.null(ttype)) stop("Unexpected transform type")

    filt <- filt.convertor(filter)
    if(is.null(filt)) stop("Unexpected filter")

    filtlen <- filtlen.calculator(filter)

    sense <- sense.convertor(direction)
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
        stop("nlevels should be an integer [whole number] between 1 and ",maxlevels)
    }

    list(x=x,xmod=xmod,len=len,sense=sense,nlevels=nlevels,ttype=ttype,filt=filt,filtlen=filtlen)
}

#' prepare.wst.arg.list
#' Prepares the arguments list for reconstruction or thresholding from a WST object
#' Internal function
#'
#' @param xwav WST object
#'
#'
prepare.wst.arg.list <- function(xwav){
    ## prepares arg.list for reconstruction or thresholding
    if(xwav$ttype == "DWT"){
        return(
            list(x = xwav$w,
                 xmod = numeric(0),
                 len = xwav$len,
                 sense = 0, ## "BWD"
                 nlevels = xwav$nlevels,
                 ttype = ttype.convertor(xwav$ttype),
                 filt = filt.convertor(xwav$filt),
                 filtlen = xwav$filtlen)
        )
    }
    if(xwav$ttype == "MODWT"){
        return(
            list(x = xwav$x,
                 xmod = xwav$w,
                 len = xwav$len,
                 sense = 0, ## "BWD"
                 nlevels = xwav$nlevels,
                 ttype = ttype.convertor(xwav$ttype),
                 filt = filt.convertor(xwav$filt),
                 filtlen = xwav$filtlen)

        )
    }
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
check.thresh.inputs <- function(xwav, hard.soft,thresh,min.level,max.level){
    arg.list <- prepare.wst.arg.list(xwav)

    if( (!is.atomic(thresh)) | (length(thresh)>1) ) stop("Only scalar thresholds supported")
    if(thresh<=0) stop("Threshold must be greater than 0")

    arg.list$thresh <- as.double(thresh)

    arg.list$sense <- NULL
    ## just removing this list element as not needed for thresholding

    arg.list$hardness <- switch(hard.soft,"hard"=0,"soft"=1)
    if(is.null(arg.list$hardness)) stop("Only hard or soft thresholding supported")

    if( (min.level!= round(min.level)) | (max.level!= round(max.level)) |
        (min.level > max.level) | (min.level <= 0) | (max.level > xwav$nlevels) ){
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

    if(arglist$sense == 1){

        if(argsin$ttype == "DWT"){
            wvt_return <- structure(list("x" = argsin$x,
                                         "ttype" = argsin$ttype,
                                         "filt" = argsin$filter,
                                         "filtlen" = arglist$filterlen,
                                         "nlevels" = arglist$nlevels,
                                         "len" = arglist$len,
                                         "w" = arglist$x),
                                    class = "WST")
        }else{
            ## MODWT
            wvt_return <- structure(list("x" = argsin$x,
                                         "ttype" = argsin$ttype,
                                         "filt" = argsin$filter,
                                         "filtlen" = arglist$filterlen,
                                         "nlevels" = arglist$nlevels,
                                         "len" = arglist$len,
                                         "w" = arglist$xmod),
                                    class = "WST")

            
        }
    }else{
        wvt_return <- arglist$x
    }
    return(wvt_return)
}

#' return.thresh
#' Puts the results into a nice structure for returning the result of thresholding.
#'
#' @param arglist Input arguments for the transform
#'
#' @export
return.thresh <- function(xwav,arg.list){
  xwav.thresh <- xwav
  if(xwav$ttype == "DWT"){
    xwav.thresh$w = arg.list$x
  }else{
    xwav.thresh$w = arg.list$xmod
  }
  return(xwav.thresh)
}

#' CPUTransform
#'
#' Wavelet transform using the CPU.
#' The input vector can be in the time domain or wavelet domain.
#' Supports DWT and (time-ordered) MODWT.
#' Supports Haar, D4, C6 and LA8 filters.
#' Allows the user to specify number of levels of transform required. Note that the maximum number of levels implemented is \eqn{\log_2{n} - b+1} where \eqn{n} is the length of the input vector and \eqn{b = ceiling (\log_2{L})} where \eqn{L} is the filter length. This means that we only allow transformations up to the level where the filter does not wrap around the coefficients more than once: with the Haar filter we do the full dyadic transform, whereas with the LA8 filter we stop after filtering 8 coefficients.
#'
#' The DWT is transformed in-place after copying the input vector, whereas the MODWT requires extra memory allocation. The structure of the transformed DWT vector is the standard interleaved form. The structure of the transformed MODWT vector is \eqn{n} scaling coefficients then detail coefficients, concatenated sequentially for each layer. The best way to access the coefficients for each level is via \code{\link{WSTtoDT}}.
#'
#' This function allows forwards or backwards transforms from a vector input for full flexibility. However, it is recommended to do the backwards transform using the Reconstruct functions which take a WST object as the only argument.
#'
#' @param xin Vector input
#' @param direction "FWD" or "BWD"
#' @param nlevels Number of levels of transform; 0 means full transform
#' @param transform.type "DWT" or "MODWT"
#' @param filter e.g. "Haar"
#'
#' @return Returns a WST object containing the transform and details.
#'
#' @seealso \code{\link{GPUTransform}}, \code{\link{CPUReconstruct}}, \code{\link{CPUThreshold}}, \code{\link{CPUSmooth}}, \code{\link{WSTtoDT}}
#'
#' @useDynLib wavecuda RcpuTransform
#' @export
CPUTransform <- function(xin, direction="FWD", nlevels=0, transform.type, filter){

    args.in <- list(ttype=transform.type,
                    filter=filter)

    if(direction == "FWD")
        args.in$x = xin
    ## if FWD transform, we add the input vector to the args.in list
    
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

#' GPUTransform
#'
#' Wavelet transform using the GPU. Note this requires a CUDA-enabled GPU.
#' The input vector can be in the time domain or wavelet domain.
#' Supports DWT and (time-ordered) MODWT.
#' Supports Haar, D4, C6 and LA8 filters.
#' Allows the user to specify number of levels of transform required. Note that the maximum number of levels implemented is \eqn{\log_2{n} - b+1} where \eqn{n} is the length of the input vector and \eqn{b = ceiling (\log_2{L})} where \eqn{L} is the filter length. This means that we only allow transformations up to the level where the filter does not wrap around the coefficients more than once: with the Haar filter we do the full dyadic transform, whereas with the LA8 filter we stop after filtering 8 coefficients.
#'
#' The DWT is transformed in-place after copying the input vector, whereas the MODWT requires extra memory allocation. The structure of the transformed DWT vector is the standard interleaved form. The structure of the transformed MODWT vector is \eqn{n} scaling coefficients then detail coefficients, concatenated sequentially for each layer. The best way to access the coefficients for each level is via \code{\link{WSTtoDT}}.
#'
#' This function allows forwards or backwards transforms from a vector input for full flexibility. However, it is recommended to do the backwards transform using the Reconstruct functions which take a WST object as the only argument.
#'
#' @param xin Vector input
#' @param direction "FWD" or "BWD"
#' @param nlevels Number of levels of transform; 0 means full transform
#' @param transform.type "DWT" or "MODWT"
#' @param filter e.g. "Haar"
#'
#' @return Returns a WST object containing the transform and details.
#'
#' @seealso \code{\link{CPUTransform}}, \code{\link{GPUReconstruct}}, \code{\link{CPUThreshold}}, \code{\link{GPUSmooth}}, \code{\link{WSTtoDT}}
#'
#' @useDynLib wavecuda RgpuTransform
#' @export
GPUTransform <- function(xin, direction, nlevels, transform.type, filter){

    args.in <- list(ttype=transform.type,
                    filter=filter)

    if(direction == "FWD")
        args.in$x = xin
    ## if FWD transform, we add the input vector to the args.in list

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

#' CPUReconstruct
#'
#' Wavelet reconstruction using the CPU.
#' The input argument is a transformed wavelet WST Object, containing all the necessary information to do a reconstruction.
#'
#' @param xwav WST object
#'
#' @return Returns a vector
#'
#' @seealso \code{\link{CPUTransform}}, \code{\link{GPUReconstruct}}, \code{\link{CPUThreshold}}, \code{\link{CPUSmooth}}
#'
#' @useDynLib wavecuda RcpuTransform
#' @export
CPUReconstruct <- function(xwav){
  arg.list <- prepare.wst.arg.list(xwav)

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

  return(arg.list$x)

}

#' GPUReconstruct
#'
#' Wavelet reconstruction using the GPU.
#' The input argument is a transformed wavelet WST Object, containing all the necessary information to do a reconstruction.
#'
#' @param xwav WST object
#'
#' @return Returns a vector
#'
#' @seealso \code{\link{GPUTransform}}, \code{\link{CPUReconstruct}}, \code{\link{CPUThreshold}}, \code{\link{CPUSmooth}}
#'
#' @useDynLib wavecuda RgpuTransform
#' @export
GPUReconstruct <- function(xwav){
  arg.list <- prepare.wst.arg.list(xwav)

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

  return(arg.list$x)

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

#' CPUThreshold
#'
#' Threshold a transformed vector using a manual threshold, with a hard or soft threshold and specifying a range of resolution levels to threshold. Uses the CPU. For further thresholding schemes, use the Smoothing functions or wavethresh.
#' Using the min and max levels, we have the flexibility to implement level-dependent thresholds.
#'
#' @param xwav WST object
#' @param thresh Threshold value
#' @param hard.soft "hard" or "soft"
#' @param min.level Min level of thresholding
#' @param max.level Max level of thresholding, the 'primary resolution'
#'
#' @seealso \code{\link{CPUSmooth}}, \code{\link{GPUSmooth}}, \code{\link{WSTtowavethresh}}
#'
#' @useDynLib wavecuda RcpuThreshold
#' @export
CPUThreshold <- function(xwav,thresh,hard.soft,min.level,max.level){
    arg.list <- check.thresh.inputs(xwav,hard.soft,thresh,min.level,max.level)

    arg.list <- .C("RcpuThreshold",
                   x=arg.list$x,
                   xmod=arg.list$xmod,
                   len=as.integer(xwav$len),
                   nlevels=as.integer(xwav$nlevels),
                   ttype=as.integer(arg.list$ttype),
                   filter=as.integer(arg.list$filt),
                   filterlen=as.integer(arg.list$filtlen),
                   thresh=arg.list$thresh,
                   hardness=as.integer(arg.list$hardness),
                   minlevel=as.integer(arg.list$min.level),
                   maxlevel=as.integer(arg.list$max.level),
                   PACKAGE="wavecuda")

    return(return.thresh(xwav,arg.list))
}

#' CPUSmooth
#'
#' Wavelet smoothing of a raw input vector. This function will perform a wavelet transform, threshold using a chosen scheme and then reconstruct. This is done on the CPU without copying the memory back and forth. Supported thresholding schemes are manual, universal and Nason's two-fold cross validation (cv). Options are hard or soft threshold, specifying a range of resolution levels to threshold and a tolerance parameter for the CV scheme.
#'
#' @param xin Vector input
#' @param nlevels Number of levels of transform; 0 means full transform
#' @param transform.type "DWT" or "MODWT"
#' @param filter e.g. "Haar"
#' @param thresh.type "manual", "univ" or "cv"
#' @param thresh Threshold value
#' @param hard.soft "hard" or "soft"
#' @param min.level Min level of thresholding
#' @param max.level Max level of thresholding, the 'primary resolution'
#' @param tol Tolerance for CV search
#'
#' @seealso \code{\link{CPUThreshold}}, \code{\link{GPUSmooth}}, \code{\link{WSTtowavethresh}}
#'
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

#' GPUSmooth
#'
#' Wavelet smoothing of a raw input vector. This function will perform a wavelet transform, threshold using a chosen scheme and then reconstruct. This is done on the GPU without copying the memory back and forth. Supported thresholding schemes are manual and Nason's two-fold cross validation (cv). NB we don't support universal thresholding as it is not efficient on the GPU. Options are hard or soft threshold, specifying a range of resolution levels to threshold and a tolerance parameter for the CV scheme.
#'
#' @param xin Vector input
#' @param nlevels Number of levels of transform; 0 means full transform
#' @param transform.type "DWT" or "MODWT"
#' @param filter e.g. "Haar"
#' @param thresh.type "manual" or "cv"
#' @param thresh Threshold value
#' @param hard.soft "hard" or "soft"
#' @param min.level Min level of thresholding
#' @param max.level Max level of thresholding, the 'primary resolution'
#' @param tol Tolerance for CV search
#'
#' @seealso \code{\link{CPUThreshold}}, \code{\link{GPUSmooth}}, \code{\link{WSTtowavethresh}}
#'
#' @useDynLib wavecuda RgpuSmooth
#' @export
GPUSmooth <- function(xin,nlevels,transform.type,filter,thresh.type,thresh=NULL,hard.soft,min.level,max.level,tol=0.01){

    arg.list <- check.smooth.inputs(xin,nlevels,transform.type,filter,thresh.type,thresh,hard.soft,min.level,max.level,tol)

    if(thresh.type=="univ") stop("Universal threshold not [yet] implemented on GPU, as it's probably quicker on CPU")

    if(filter %in% c("D4", "C6", "LA8")) stop("Other filters not yet implemented for smoothing")
    
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

## to be added - GPU transform list to efficiently transform a list of vectors on the GPU
## to make use of streams
## #' @useDynLib wavecuda RgpuTransformList
## #' @export
## GPUTransformList <- function(xin, direction, nlevels, transform.type, filter){
##     ## xin should be a list
##     ## direction, nlevels, transform.type, filter should be vectors
##     ## all of the same length...

##     len <- length(xin)

##     if(len<=1) stop("We require a list of xin vectors of length > 1")

##     if( (length(direction) != len) || (length(nlevels) != len) || (length(transform.type) != len) || (length(filter) != len) ) stop("Inconsistent length of input list & vectors")

##     arg.list <- list()

##     for(il in 1:len){
##         arg.list[[il]] <- check.trans.inputs(xin[[il]], direction[il], nlevels[il], transform.type[il], filter[il])
##     }

##     ret.list <- .Call("RgpuTransformList",
##                       arglist <- arg.list,
##                       package="wavecuda")

##     print("Done....")

##     return(ret.list)

## }

## Nason's CV, with corrected interpolation
## ## #' @export
## wstCV1 <- function (ndata, ll = 3, type = "soft", filter.number = 10, family = "DaubLeAsymm",
##     tol = 0.01, verbose = 0, plot.it = FALSE, norm = l2norm,
##     InverseType = "average", uvdev = madmad)
## {
##     ## edit of Nason's wstCV for diagnostics
##     nlev <- log(length(ndata))/log(2)
##     levels <- ll:(nlev - 1)
##     nwst <- wst(ndata, filter.number = filter.number, family = family)
##     uv <- threshold(nwst, levels = levels, type = type, policy = "universal",
##         dev = madmad, return.thresh = TRUE)[1]
##     if (verbose == 1)
##         cat("Now optimising cross-validated error estimate\n")
##     levels <- ll:(nlev - 2)
##     R <- 0.61803399
##     C <- 1 - R
##     ax <- 0
##     bx <- uv/2
##     cx <- uv
##     x0 <- ax
##     x3 <- cx
##     if (abs(cx - bx) > abs(bx - ax)) {
##         x1 <- bx
##         x2 <- bx + C * (cx - bx)
##     }
##     else {
##         x2 <- bx
##         x1 <- bx - C * (bx - ax)
##     }
##     fa <- GetRSSWST(ndata, threshold = ax, levels = levels, type = type,
##         filter.number = filter.number, family = family, norm = norm,
##         verbose = verbose, InverseType = InverseType)
##     ## cat("Done 1\n")
##     fb <- GetRSSWST(ndata, threshold = bx, levels = levels, type = type,
##         filter.number = filter.number, family = family, norm = norm,
##         verbose = verbose, InverseType = InverseType)
##     ## cat("Done 2\n")
##     fc <- GetRSSWST(ndata, threshold = cx, levels = levels, type = type,
##         filter.number = filter.number, family = family, norm = norm,
##         verbose = verbose, InverseType = InverseType)
##     ## cat("Done 3\n")
##     f1 <- GetRSSWST(ndata, threshold = x1, levels = levels, type = type,
##         filter.number = filter.number, family = family, norm = norm,
##         verbose = verbose, InverseType = InverseType)
##     ## cat("Done 4\n")
##     f2 <- GetRSSWST(ndata, threshold = x2, levels = levels, type = type,
##         filter.number = filter.number, family = family, norm = norm,
##         verbose = verbose, InverseType = InverseType)
##     ## cat("Done 5\n")
##     xkeep <- c(ax, cx, x1, x2)
##     fkeep <- c(fa, fc, f1, f2)
##     if (plot.it == TRUE) {
##         plot(c(ax, bx, cx), c(fa, fb, fc))
##         text(c(x1, x2), c(f1, f2), lab = c("1", "2"))
##     }
##     cnt <- 3
##     while (abs(x3 - x0) > tol * (abs(x1) + abs(x2))) {
##         if (verbose > 0) {
##             cat("x0=", x0, "x1=", x1, "x2=", x2, "x3=", x3, "\n")
##             cat("f1=", f1, "f2=", f2, "\n")
##         }
##         if (f2 < f1) {
##             x0 <- x1
##             x1 <- x2
##             x2 <- R * x1 + C * x3
##             f1 <- f2
##             f2 <- GetRSSWST(ndata, threshold = x2, levels = levels,
##                 type = type, filter.number = filter.number, family = family,
##                 norm = norm, verbose = verbose, InverseType = InverseType)
##             if (verbose == 2) {
##                 cat("SSQ: ", signif(f2, digits = 3), "\n")
##             }
##             else if (verbose == 1)
##                 cat(".")
##             xkeep <- c(xkeep, x2)
##             fkeep <- c(fkeep, f2)
##             if (plot.it == TRUE)
##                 text(x2, f2, lab = as.character(cnt))
##             cnt <- cnt + 1
##         }
##         else {
##             x3 <- x2
##             x2 <- x1
##             x1 <- R * x2 + C * x0
##             f2 <- f1
##             f1 <- GetRSSWST(ndata, threshold = x1, levels = levels,
##                 type = type, filter.number = filter.number, family = family,
##                 norm = norm, verbose = verbose, InverseType = InverseType)
##             if (verbose == 2)
##                 cat("SSQ: ", signif(f1, digits = 3), "\n")
##             else if (verbose == 1)
##                 cat(".")
##             xkeep <- c(xkeep, x1)
##             fkeep <- c(fkeep, f1)
##             if (plot.it == TRUE)
##                 text(x1, f1, lab = as.character(cnt))
##             cnt <- cnt + 1
##         }
##     }
##     if (f1 < f2)
##         tmp <- x1
##     else tmp <- x2
##     x1 <- tmp/sqrt(1 - log(2)/log(length(ndata)))
##     if (verbose == 1)
##         cat("Correcting to ", x1, "\n")
##     else if (verbose == 1)
##         cat("\n")
##     g <- sort.list(xkeep)
##     xkeep <- xkeep[g]
##     fkeep <- fkeep[g]
##     if (verbose >= 1) {
##         cat("Reconstructing CV \n")
##     }
##     nwstT <- threshold(nwst, type = type, levels = levels, policy = "manual",
##         value = x1)
##     nwstT <- threshold(nwstT, type = type, levels = nlevelsWT(nwstT) -
##         1, policy = "universal", dev = uvdev)
##     ## not sure why he does an extra universal threshold here!
##     xvwr <- AvBasis.wst(nwstT)
##     list(ndata = ndata, xvwr = xvwr, xvwrWSTt = nwstT, uvt = uv,
##         xvthresh = x1, xkeep = xkeep, fkeep = fkeep)
## }

#' print.WST
#' Print function for the WST object. Prints a brief summary of the wavelet transform.
#'
#' @param xwav WST object
#'
#'@export
print.WST <- function(xwav){
    cat("--------------------------------------\n")
    cat("Wavecuda STructure object WST class")
    cat("\n of type:                  ",xwav$ttype)
    cat("\n with filter:              ",xwav$filt)
    cat("\n levels of transform:      ",xwav$nlevels)
    cat("\n original vector of length:",xwav$len)
    cat("\n")
    cat("--------------------------------------\n")
}

WSTtoDTold <- function(Xwav, scaling = TRUE, forPlotting = FALSE){
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
    padct <- rep("d", Xwav$nlevels * 2) ## coefftype

    if(Xwav$ttype == "DWT"){
        ## sep level list for d and s
        ## s level list is just for top level
        ## ll <- list()
        ## ll[[Xwav$nlevels]] <- getCoeffLevel(Xwav,Xwav$nlevels,"s")
        xw_df <- data.table(W = c(Xwav$w, padNA),
                            Level = c(makeLevelNVec(Xwav,levelList), padl),
                            Translate = c(makeLevelTVec(Xwav,levelList), pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$w)),padmm)
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
        xw_df <- data.table(W = c(Xwav$w[detailCoeffSelector], padNA),
                            Level = c(makeLevelNVec(Xwav,levelList),padl),
                            Translate = c(makeLevelTVec(Xwav, levelList),pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$w)/2),padmm)
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
    ## =====  NB : old filter for scaling coeffs  ======

    if(!forPlotting){
        xw_df <- xw_df[minmax == 0]
        ## remove min max extra values
        xw_df[,minmax := NULL]
        ## remove min max column
    }

    return(xw_df)
}

#' WSTtoWavethresh
#'
#' Converts a WST object to a wavethresh object to use the wavethresh library. Three things to note on design differences between wavecuda and wavethresh:
#' - WaveCUDA does not retain scaling coefficients not required for the transform but wavethresh does
#' - wavethresh does not support the C6 wavelet
#' - wavethresh uses a different ordering for the (time-ordered) MODWT coefficients
#'
#' @param XW WST object
#' @param showWarnings Boolean option to show warnings on incompatibilities in conversions
#'
#' @import data.table
#' @import wavethresh
#' @export
WSTtoWavethresh <- function(XW, showWarnings = TRUE){
    xw_df <- WSTtoDT(XW)
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
                              v = xw_df[(Level == l) & (CoeffType == "d"), W])
    }
    XW_wavethresh <- putC(XW_wavethresh,
                          level = XW_wavethresh$nlevels - l,
                          v = xw_df[(Level == l) & (CoeffType == "s"), W])

    return(XW_wavethresh)
}

#' label.detail.scaling
#'
#' Returns a vector of "d" and "s" corresponding to detail and scaling labels for the coefficients.
#' Internal function.
#'
#'@param Xwav WST object
#'
label.detail.scaling <- function(Xwav){
    if(Xwav$ttype == "DWT"){
        lab <- ifelse(((1:Xwav$len) %% 2^(Xwav$nlevels) == 1),
                      "s",
                      "d")
    }
    if(Xwav$ttype == "MODWT"){
        lab <- rep(rep(c("s","d"),each = Xwav$len),times = Xwav$nlevels)
    }
    return(lab)
}

#' WSTtoDT
#'
#' Converts a WST object to a data.table, recording time (translate) and level information for each wavelet coefficient. This takes the wavelet vector out of its 'raw' form where the coefficients are unlabelled and even interleaved for the DWT, and puts them in a more user-friendly form for performing further analysis on wavelet coefficients.
#' Used also for plotting.
#'
#' @param Xwav WST object
#' @param scaling boolean determining whether we retain scaling coefficients (default TRUE)
#' @param forPlotting boolean determining whether we keep the min/max values (default FALSE)
#'
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
    padct <- rep("d", Xwav$nlevels * 2) ## coefftype

    if(Xwav$ttype == "DWT"){
        xw_df <- data.table(W = c(Xwav$w, padNA),
                            CoeffType = c(label.detail.scaling(Xwav),padct),
                            Level = c(rep(1,Xwav$len),padl),
                            Translate = c(rep(seq(0.5,Xwav$len/2,1),each=2),pad0),
                            ## we initialise the values for level & translate
                            ## they should be valid for half or all points
                            minmax = c(rep(0,length(Xwav$w)),padmm))
        
        if(Xwav$nlevels > 1){
            ## if nlevels > 1, then we need to overwrite
            ## levels and translates
            for(l in (2:Xwav$nlevels)){
                ## we loop through the levels for the details
                levelRowIDs <- getCoeffLevel(Xwav,l,"d")
                ## we write the updated level
                xw_df[levelRowIDs, Level := l]
                ## we write the updated translate
                translates <- seq(from = 2^(l-2), to = Xwav$len/2, by = 2^(l-1))
                xw_df[levelRowIDs, Translate := translates]
            }

            ## then we add the scalings
            levelRowIDs <- getCoeffLevel(Xwav,l,"s")
            xw_df[levelRowIDs, Level := l]
            xw_df[levelRowIDs, Translate := translates]
        }

        ## now just functionalise
    }

    ## add MODWT version. Should be much easier!
    ## translate is c(rep(1:len, 2 * nlevels), pad)
    ## level is c(rep(1:nlevels, len), pad)

    if(Xwav$ttype == "MODWT"){
        xw_df <- data.table(W = c(Xwav$w, padNA),
                            CoeffType = c(label.detail.scaling(Xwav),padct),
                            Level = c(rep(1:Xwav$nlevels,each=2*Xwav$len),padl),
                            Translate = c(rep(seq(1:Xwav$len),2*Xwav$nlevels),pad0),
                            minmax = c(rep(0,length(Xwav$w)),padmm))
    }
    ## sort the min/max values (for plotting)
    xw_df[(CoeffType == "d"), W := replace(W, (Translate == 0), max(abs(W), na.rm = T)),
          by = c("Level")]
    ## replace the created NA values with max abs per level
    xw_df[(Translate == 0) & (minmax == 1), W:=-W]
    ## set the min vals per level

    if(!forPlotting){
        xw_df <- xw_df[minmax == 0]
        ## remove min max extra values
        xw_df[,minmax := NULL]
        ## remove min max column
    }

    if(!scaling){
        xw_df <- xw_df[CoeffType == "d"]
        ## filter to only the detail coeffs
    }
    
    return(xw_df)
}
