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

#' @export
makeLevelNVec <- function(XW,levelList){
    if(XW$ttype == "DWT"){
    lnv <- rep(0,XW$len)
    levels = XW$nlevels
    for(l in 1:levels)
        lnv[levelList[[l]]] <- l
    }
    if(substr(XW$ttype,1,5) == "MODWT"){
        lnv <- rep(1:levels, each = XW$len)
    }
    lnv
}

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

#' @export
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

#' @export
plotDWT <- function(Xwav, method = "dp"){
    levelList <- sapply(X = 1:Xwav$nlevels, FUN = function(l) getCoeffLevel(Xwav,l,"d"), simplify = F)

    if(method == "dt"){
        ## placeholders for min/max treatment
        ## because we can't plot y_free with symmetric axes
        ## so we create dummy rows at the end of the data frame
        ## all with translate value of 0
        ## and initialise these with NA for W
        padNA <- rep(NA, Xwav$nlevels * 2)
        pad0 <- rep(0, Xwav$nlevels * 2)
        padl <- rep(1:Xwav$nlevels, each = 2) # levels
        padmm <- rep(1:2, Xwav$nlevels) # minmax
        
        xw_df <- data.table(W = c(Xwav$x, padNA),
                            Level = c(makeLevelNVec(Xwav,levelList), padl),
                            Translate = c(makeLevelTVec(Xwav,levelList), pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$x)),padmm)
                            ## 0 for values of the transform
                            ## 1 for  min(-abs(W))
                            ## 2 for  max(abs(W)) per level
                            )
        xw_df[,T2 := ifelse(.I <= Wxwav$len, (.I-1)/2,0)]
        xw_df[,L2 := quickLVec(.I)]
        xw_df[, W := replace(W, Translate == 0, max(abs(W), na.rm = T)),
              by = c("Level")]
        ## replace the created NA values with max abs per level
        xw_df[(Translate == 0) & (minmax == 1), W:=-W]
        ## set the min vals per level
        xw_df <- xw_df[Level >0]
        ## filter -> will be done below
    }

    if(method == "dp"){
        xw_df <- data.frame(W = Xwav$x,
                            Level = makeLevelNVec(Xwav,levelList),
                            Translate = makeLevelTVec(Xwav,levelList)) %>%
            mutate(minmax = 0)
        ## form a DF of the wavelet decomp
        
        xMinMax <- xw_df %>%
            group_by(Level) %>%
            summarise(Wmax = max(abs(W))) %>%
            mutate(Wmin = -Wmax) %>%
            gather(minmax,W,Wmax:Wmin) %>%
            mutate(minmax = ifelse(minmax == "Wmax",2,1)) %>%
            mutate(Translate = 0)
        ## calculate min/max for each level
        ## for the purpose of plotting
        ## (as we can't seem to plot with symmetric & free y axes)

        xw_df <- xw_df %>%
            rbind(xMinMax) %>%
            filter(Level>0)
    }
    
    ## now we plot with ggplot...
    ggplot(data = xw_df, mapping=aes(x = Translate, y = W, fill = as.factor(minmax))) +
        geom_bar(stat = "identity") +
        ## plot bars
        scale_fill_manual(breaks = c("0","1","2"), values=c(rgb(0,0,0,1), rgb(1,1,1,0), rgb(1,1,1,0)),guide=FALSE ) +
        ## this colours the dummy min/max values as transparent and the rest as black
        facet_grid(Level ~ ., scales = "free_y", as.table = F, labeller = label_both) +
        ## no free & symmetric y option
        scale_x_continuous(breaks=seq(0, Xwav$len/2, Xwav$len/8 )) +
        ## just label 5 points 
        labs(title = paste("Wavelet Decomposition,",Xwav$filt,Xwav$ttype), y = "Wavelet Coefficients")
}

#' @export
plotDWTempty <- function(Xwav){
    levelList <- sapply(X = 1:Xwav$nlevels, FUN = function(l) getCoeffLevel(Xwav,l,"d"), simplify = F)
    
    ## Level = emptyFunction(levelList)
    ## Translate = emptyFunction(levelList)
    return(0)
}


emptyFunction <- function(mylist){
    return(0)
}

#' @export
plotMODWT <- function(Xwav, method = "dp"){
    levelList <- sapply(X = 1:Xwav$nlevels, FUN = function(l) getCoeffLevel(Xwav,l,"d"),simplify=F)

    if(method == "dt"){
        ## placeholders for min/max treatment
        ## because we can't plot y_free with symmetric axes
        ## so we create dummy rows at the end of the data frame
        ## all with translate value of 0
        ## and initialise these with NA for W
        padNA <- rep(NA, Xwav$nlevels * 2)
        pad0 <- rep(0, Xwav$nlevels * 2)
        padl <- rep(1:Xwav$nlevels, each = 2) # levels
        padmm <- rep(1:2, Xwav$nlevels) # minmax

        detailCoeffSelector <- (1:Xwav$len)*
        
        xw_df <- data.table(W = c(Xwav$xmod, padNA),
                            Level = c(makeLevelNVec(Xwav,levelList),padl),
                            Translate = c(makeLevelTVec(Xwav, levelList),pad0),
                            ## time component of the coef through the transform
                            minmax = c(rep(0,length(Xwav$xmod)),padmm)
                            ## 0 for values of the transform
                            ## 1 for  min(-abs(W))
                            ## 2 for  max(abs(W)) per level
                            )
        xw_df[, W := replace(W, Translate == 0, max(abs(W), na.rm = T)),
              by = c("Level")]
        ## replace the created NA values with max abs per level
        xw_df[(Translate == 0) & (minmax == 1), W:=-W]
        ## set the min vals per level
        xw_df <- xw_df[Level >0]
        ## filter -> will be done below    
    }
    
    if(method == "dp"){
        xw_df <- data.frame(W = Xwav$xmod,
                            CoeffType = rep(rep(c(0,1),each = Xwav$len),Xwav$nlevels)) %>%
            filter(CoeffType == 1) %>%
            mutate(Level = makeLevelNVec(Xwav,levelList),
                   Translate = makeLevelTVec(Xwav,levelList)) %>%
            filter(Level > 0) %>%
            mutate(minmax = 0) %>%
            select(-CoeffType)
        ## form a DF of the wavelet decomp

        xMinMax <- xw_df %>%
            group_by(Level) %>%
            summarise(Wmax = max(abs(W))) %>%
            mutate(Wmin = -Wmax) %>%
            gather(minmax,W,Wmax:Wmin) %>%
            mutate(minmax = ifelse(minmax == "Wmax",2,1)) %>%
            mutate(Translate = 0)
        ## calculate min/max for each level
        ## for the purpose of plotting
        ## (as we can't seem to plot with symmetric & free y axes)

        xw_df <- xw_df %>%
            rbind(xMinMax)
    }

    
    ## now we plot with ggplot...
    ggplot(data = xw_df, mapping=aes(x = Translate, y = W, fill = as.factor(minmax))) +
        geom_bar(stat = "identity") +
        scale_fill_manual(breaks = c("0","1","2"), values=c(rgb(0,0,0,1), rgb(1,1,1,0), rgb(1,1,1,0)),guide=FALSE ) +
        facet_grid(Level ~ ., scales = "free_y", as.table = F, labeller = label_both) +
        scale_x_continuous(breaks=seq(0, Xwav$len, Xwav$len/8 )) +
        labs(title = paste("Wavelet Decomposition,",Xwav$filt,Xwav$ttype), y = "Wavelet Coefficients")    
}



