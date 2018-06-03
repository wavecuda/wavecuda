
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
    cat("Done 1\n")
    fb <- GetRSSWST(ndata, threshold = bx, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    cat("Done 2\n")
    fc <- GetRSSWST(ndata, threshold = cx, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    cat("Done 3\n")
    f1 <- GetRSSWST(ndata, threshold = x1, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    cat("Done 4\n")
    f2 <- GetRSSWST(ndata, threshold = x2, levels = levels, type = type, 
        filter.number = filter.number, family = family, norm = norm, 
        verbose = verbose, InverseType = InverseType)
    cat("Done 5\n")
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
    ## nwstT <- threshold(nwstT, type = type, levels = nlevelsWT(nwstT) - 
    ##     1, policy = "universal", dev = uvdev)
    ## not sure why he does an extra universal threshold here!
    xvwr <- AvBasis.wst(nwstT)
    list(ndata = ndata, xvwr = xvwr, xvwrWSTt = nwstT, uvt = uv, 
        xvthresh = x1, xkeep = xkeep, fkeep = fkeep)
}


## big test to come...

library(wmtsa)

signame <- "doppler"
len <- 128

## clean
X <- make.signal(signame,n=len,snr=Inf)
dop <- X@data

wdop <- CPUTransform(dop,"FWD",5,"DWT","D4")

rdop <- CPUTransform(wdop,"BWD",5,"DWT","D4")

tdop <- CPUThreshold(wdop,5,"DWT","D4",0.5,"hard",1,3)

wdop2 <- GPUTransform(dop,"FWD",5,"DWT","Haar")

rdop2 <- GPUTransform(wdop2,"BWD",5,"DWT","Haar")


sdop <- CPUSmooth(xin=dop,nlevels=5,transform.type="DWT",filter="D4",thresh.type="manual",thresh=0.5,hard.soft="hard",min.level=1,max.level=3)

sdop <- CPUSmooth(xin=dop,nlevels=6,transform.type="DWT",filter="D4",thresh.type="univ",hard.soft="hard",min.level=1,max.level=6)

sdop1 <- CPUSmooth(xin=dop,nlevels=6,transform.type="DWT",filter="D4",thresh.type="univ",hard.soft="hard",min.level=1,max.level=1)

sdop2 <- CPUSmooth(xin=dop,nlevels=7,transform.type="DWT",filter="Haar",thresh.type="univ",hard.soft="hard",min.level=1,max.level=7)

sdop3 <- CPUSmooth(xin=dop,nlevels=7,transform.type="DWT",filter="Haar",thresh.type="univ",hard.soft="hard",min.level=1,max.level=1)

sdop4 <- CPUSmooth(xin=dop,nlevels=7,transform.type="DWT",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=4)

sdop5 <- GPUSmooth(xin=dop,nlevels=7,transform.type="DWT",filter="Haar",thresh.type="manual",thresh=0.2,hard.soft="hard",min.level=1,max.level=4)

sdop6 <- GPUSmooth(xin=dop,nlevels=7,transform.type="DWT",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=4)


## sort out sense variable. Maybe create a nice wavelet structure?
## need to check minlevel, maxlevel, nlevels, filtlen
## length value for MODWT
## NB have made edits to Makefile & a few code files! @@@ Version control alarm! @@@



## CV timings!!

library(wavethresh)

### DWT first.........

J <- 24
n <- 2^J

X <- make.signal(signame,n=n,snr=5)
bdop <- X@data

system.time(wc.cpu.dwt <- CPUSmooth(xin=bdop,nlevels=J,transform.type="DWT",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=J))


system.time(wc.gpu.dwt <- GPUSmooth(xin=bdop,nlevels=J,transform.type="DWT",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=J))


system.time(nas.dwt <- CWCV(bdop,ll=0,filter.number=1,family="DaubExPhase",thresh.type="soft",interptype="normal",plot.it=FALSE))

plot(wc.cpu.dwt,type="l")
points(wc.gpu.dwt,type="l",col=2)
points(nas.dwt$xvwr,type="l",col=3)


X <- make.signal(signame,n=n,snr=Inf)
truedop <- X@data


sum((truedop - nas.dwt$xvwr)^2)/n
sum((truedop - wc.cpu.dwt)^2)/n
sum((truedop - wc.gpu.dwt)^2)/n


### MODWT next.........

J <- 19
n <- 2^J

X <- make.signal(signame,n=n,snr=5)
bdop <- X@data

system.time(wc.cpu.modwt <- CPUSmooth(xin=bdop,nlevels=J,transform.type="MODWT.TO",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=J))


system.time(wc.gpu.modwt <- GPUSmooth(xin=bdop,nlevels=J,transform.type="MODWT.TO",filter="Haar",thresh.type="cv",hard.soft="soft",min.level=1,max.level=J))


system.time(nas.modwt <- wstCV(bdop,ll=0,filter.number=1,family="DaubExPhase",type="soft",verbose=TRUE))

plot(wc.cpu.modwt,type="l")
points(wc.gpu.modwt,type="l",col=2)
points(nas.modwt$xvwr,type="l",col=3)

## Nason's MODWT version is beautiful...he must have done something differently!

mwtdop <- CPUSmooth(bdop,J,"MODWT.PO","Haar","manual",nas.modwt$xvthresh,"soft",1,J)

## we get the same as our result when we reconstruct after thresholding with his thresh
## So I think it might actually come from the MODWT code itself.

## double checking his...

## if we threshold using normal functions & avbasis reconstruction,
## then we get the same reconstruction as our code
## so it's just his wstCV code that does it differently

wdp <- wd(bdop,filter.number=1,family="DaubExPhase",type="station")
wdp <- convert(wdp)
wdpt <- threshold(wdp,levels=(0:(J-1)),policy="manual",value=nas.modwt$xvthresh)
rdp <- AvBasis(wdpt)

nas.modwt1 <- wstCV1(bdop,ll=0,filter.number=1,family="DaubExPhase",type="soft",verbose=TRUE)

X <- make.signal(signame,n=n,snr=Inf)
truedop <- X@data

sum((truedop - nas.modwt$xvwr)^2)/n
sum((truedop - wc.cpu.modwt)^2)/n
sum((truedop - wc.gpu.modwt)^2)/n
sum((truedop - nas.modwt1$xvwr)^2)/n
