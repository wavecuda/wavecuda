#' getCoeffLevel
#' Get wavelet or scaling coefficients at a given level
#' Internal function.
#'
#' @param xw WST object
#' @param level Level of transform
#' @param coeffType Detail or scaling coefficient
#'
#' @export
getCoeffLevel <- function(xw, level, coeffType = "d"){
    ## see isolate_dlevels in wavutils.cpp
    ## NB higher level means deeper wvt coeffs

    if(class(xw) != "WST")
        stop("WST type required")

    ## check level <= level of transform
    if((0 >= level)|(level>xw$nlevels)|level != as.integer(level))
        stop("Integer level expected, between 1 and ", xw$nlevels)

    ## type = DWT

    ## if DWT, also check that scaling coeffs available
    ## only if requested number == level

    stopifnot((coeffType == "d" | coeffType == "s"))
    d_bool = (coeffType == "d") ## detail or scaling
    n = xw$len

    if(xw$ttype == "DWT"){
        if(!d_bool){
            stopifnot(level == xw$nlevels)
        }

        start = 2^(level-1)*d_bool+ 1
        gap_i = 2^(level)
    }

    if(substr(xw$ttype,1,5) == "MODWT"){
        start = 1
        gap_i = 1
    }

    dlevel_i = seq(from = start, to = n, by = gap_i)
    return(dlevel_i)
}

#' makeLevelNVec
#' Makes a list indicating length of coefficient vector for each level of transform.
#' Internal function
#'
#' @param XW WST object
#' @param levelList List of levels of transform
#'
#' @export
makeLevelNVec <- function(XW,levelList){
    if(XW$ttype == "DWT"){
        lnv <- rep(0,XW$len)
        for(l in 1:XW$nlevels)
            lnv[levelList[[l]]] <- l
    }
    if(substr(XW$ttype,1,5) == "MODWT"){
        lnv <- rep(1:XW$nlevels, each = XW$len)
    }
    lnv
}

#' makeLevelTVec
#' Makes a list of 'translate' values for each wavelet coefficient.
#' i.e. time value corresponding to wavelet coefficients through each level.
#'
#' @param XW WST object
#' @param levelList List of levels
#'
#' @export
makeLevelTVec <- function(XW,levelList){
    ## assuming full transform...
    if(XW$ttype == "DWT"){
        ltv <- rep(0,XW$len)
        levels = XW$nlevels
        for(l in 1:levels)
            ltv[levelList[[l]]] <- seq(from = 2^(l-2), to = XW$len/2, by = 2^(l-1))
    }
    if(substr(XW$ttype,1,5) == "MODWT"){
        ltv <- unlist(levelList)
    }
    ltv
}

## export
## WIP to get LVec more efficiently
quickLVec <- function(RN, maxlevel){
    ## wrong!!
    if(RN == 1)
        return(maxlevel)
    if( RN %% 2 == 0)
        return(1)
    l <- 1
    while( (RN %% (2^l)) > 1){
        l = l+1
    }
    if( (RN %% (2^(l+1))) > 1)
        return(l)
    else return(l+1)
}

#' signifDown
#' Rounds down to a specified number of signficant figures.
#' Internal function, used for plotting labels.
#'
#' @param x Number to round
#' @param digits Number of significant figures
#'
#' @export
signifDown <- function(x, digits){
    m <- 10^floor(log10(abs(x))-(digits - 1))
    ## 'round' is necessary in case of
    ## small errors from floating point accuracy
    xSignifDown <- floor(round(abs(x)/m,3))*m
    return(sign(x)*xSignifDown)
}

#' Plotting for WST objects
#' Plots the wavelet transform at all levels using ggplot.
#'
#' @param Xwav WST object
#'
#' @import ggplot2
#' @export
plot.WST <- function(Xwav){
    xw_df <- WSTtoDT(Xwav, forPlotting = TRUE)
    ## turn WST object into data table

    ## now we plot with ggplot...
    p <- ggplot(data = xw_df, mapping=aes(x = Translate, y = W, fill = as.factor(minmax))) +
        ## minmax ~ plus/minus abs max
        geom_bar(stat = "identity") +
        scale_fill_manual(breaks = c("0","1","2"), values=c(rgb(0,0,0,1), rgb(1,1,1,0), rgb(1,1,1,0)),guide=FALSE ) +
        ## plot the min/max values as transparent
        scale_y_continuous(breaks = function(lims) return(c(signifDown(lims[1],1),0,signifDown(lims[2],1)))) +
        ## adds 3 labels to the y axis, at signifDown(min,1), signifDown(max,1) and 0
        ## where signifDown rounds down to 1 s.f.
        ## (if we rounded up, the axis label wouldn't be visible)
        facet_grid(Level ~ ., scales = "free_y", as.table = F, labeller = label_both) +
        ## free y axis scale allows us to scale by level
        ## and the min/max ensures symmetric axes
        scale_x_continuous(breaks=seq(0, Xwav$len, Xwav$len/8 )) +
        labs(title = paste("Wavelet Decomposition,",Xwav$filt,Xwav$ttype), y = "Wavelet Coefficients")

    p
    return(p)
}
