Homework_4
================
Ningyu Han
2023-02-27

``` r
library('MASS') ## for 'mcycle'
library('manipulate') ## for 'manipulate'
library('manipulate')
library('splines') ## 'ns'
library('caret') ## 'knnreg' and 'createFolds'
```

    ## Loading required package: ggplot2

    ## Loading required package: lattice

``` r
y <- mcycle$accel
x <- matrix(mcycle$times, length(mcycle$times), 1)

plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
```

![](Homework_4_files/figure-gfm/unnamed-chunk-2-1.png)<!-- -->

## Randomly split the mcycle data into training (75%) and validation (25%) subsets.

``` r
dat <- mcycle
set.seed(123)

n <- nrow(dat)
train_indices <- sample(1:n, round(0.75*n), replace = FALSE)
mcycle_train <- mcycle[train_indices, ]
mcycle_test <- mcycle[-train_indices, ]
```

``` r
y_valid <- mcycle_test$accel
x_valid <- matrix(mcycle_test$times, length(mcycle_test$times), 1)

plot(x_valid, y_valid, xlab="Time (ms)", ylab="Acceleration (g)")
```

![](Homework_4_files/figure-gfm/unnamed-chunk-4-1.png)<!-- -->

## Using the mcycle data, consider predicting the mean acceleration as a function of time. Use the Nadaraya-Watson method with the k-NN kernel function to create a series of prediction models by varying the tuning parameter over a sequence of values. (hint: the script already implements this)

``` r
## Epanechnikov kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## lambda - bandwidth (neighborhood size)
kernel_epanechnikov <- function(x, x0, lambda=1) {
  d <- function(t)
    ifelse(t <= 1, 3/4*(1-t^2), 0)
  z <- t(t(x) - x0)
  d(sqrt(rowSums(z*z))/lambda)
}
```

``` r
## k-NN kernel function
## x  - n x p matrix of training inputs
## x0 - 1 x p input where to make prediction
## k  - number of nearest neighbors
kernel_k_nearest_neighbors <- function(x, x0, knn=1) {
  ## compute distance betwen each x and x0
  z <- t(t(x) - x0)
  d <- sqrt(rowSums(z*z))

  ## initialize kernel weights to zero
  w <- rep(0, length(d))
  
  ## set weight to 1 for k nearest neighbors
  w[order(d)[1:knn]] <- 1
  
  return(w)
}
```

``` r
## Make predictions using the NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## x0 - m x p matrix where to make predictions
## kern  - kernel function to use
## ... - arguments to pass to kernel function
nadaraya_watson <- function(y, x, x0, kern, ...) {
  k <- t(apply(x0, 1, function(x0_) {
    k_ <- kern(x, x0_, ...)
    k_/sum(k_)
  }))
  yhat <- drop(k %*% y)
  attr(yhat, 'k') <- k
  return(yhat)
}

yhat <- nadaraya_watson(y, x, x,kernel_k_nearest_neighbors, knn=1)
```

``` r
## Helper function to view kernel (smoother) matrix
matrix_image <- function(x) {
  rot <- function(x) t(apply(x, 2, rev))
  cls <- rev(gray.colors(20, end=1))
  image(rot(x), col=cls, axes=FALSE)
  xlb <- pretty(1:ncol(x))
  xat <- (xlb-0.5)/ncol(x)
  ylb <- pretty(1:nrow(x))
  yat <- (ylb-0.5)/nrow(x)
  axis(3, at=xat, labels=xlb)
  axis(2, at=yat, labels=ylb)
  mtext('Rows', 2, 3)
  mtext('Columns', 3, 3)
}

matrix_image(attr(yhat, 'k'))
```

![](Homework_4_files/figure-gfm/unnamed-chunk-8-1.png)<!-- -->

## With the squared-error loss function, compute and plot the training error, AIC, BIC, and validation error (using the validation data) as functions of the tuning parameter.

``` r
## Compute effective df using NW method
## y  - n x 1 vector of training outputs
## x  - n x p matrix of training inputs
## kern  - kernel function to use
## ... - arguments to pass to kernel function
effective_df <- function(y, x, kern, ...) {
  y_hat <- nadaraya_watson(y, x, x,
    kern=kern, ...)
  sum(diag(attr(y_hat, 'k')))
}
```

``` r
x_train = matrix(mcycle_train$times, length(mcycle_train$times), 1)
yhat_train = nadaraya_watson(mcycle_train$accel, x_train, x_train, kernel_k_nearest_neighbors, knn=1)
```

``` r
## loss function
## y    - train/test y
## yhat - predictions at train/test x
  
loss_squared_error <- function(y, yhat)
  (y - yhat)^2

## test/train error
## y    - train/test y
## yhat - predictions at train/test x
## loss - loss function
error <- function(y, yhat, loss=loss_squared_error)
  mean(loss(y, yhat))

train_error <- rep(NA, 20)
for (i in 1:20){
  yhat_train <- nadaraya_watson(mcycle_train$accel, x_train, x_train, kernel_k_nearest_neighbors, knn = i)
  train_error[i] <- error(mcycle_train$accel, yhat_train, loss = loss_squared_error)
}

plot(seq(1:20),train_error, xlab = "k", ylab = "training error")
```

![](Homework_4_files/figure-gfm/unnamed-chunk-11-1.png)<!-- -->

``` r
y = mcycle_train$accel
x <- x_train

## AIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom
aic <- function(y, yhat, d)
  error(y, yhat) + 2/length(y)*d

## BIC
## y    - training y
## yhat - predictions at training x
## d    - effective degrees of freedom

bic <- function(y, yhat, d)
  error(y, yhat) + log(length(y))/length(y)*d

## make predictions using NW method at training inputs
yhat <- nadaraya_watson(y, x, x,
  kernel_epanechnikov, lambda=5)

## view kernel (smoother) matrix
# matrix_image(attr(yhat, 'k'))

## compute effective degrees of freedom
edf <- effective_df(y, x, kernel_epanechnikov, lambda=5)
aic(y, yhat, edf)
```

    ## [1] 722.7298

``` r
bic(y, yhat, edf)
```

    ## [1] 722.9516

``` r
aic_error <- rep(NA, 20)
for(i in 1:20){
  yhat_train <- nadaraya_watson(mcycle_train$accel, x_train, x_train,kernel_k_nearest_neighbors, knn=i)
  edf <- effective_df(y, x, kernel_k_nearest_neighbors, knn=i)
  aic_error[i] <-  aic(y, yhat_train, edf)
}
print(aic_error)
```

    ##  [1] 365.0155 427.9374 502.4882 489.7461 503.1019 543.6581 555.7789 543.9533
    ##  [9] 560.7425 584.5709 581.8404 586.9614 601.9214 609.4732 617.3089 637.4574
    ## [17] 658.3733 671.1092 685.2458 718.8412

``` r
plot(seq(1:20), aic_error, xlab = 'k',ylab = 'AIC')
```

![](Homework_4_files/figure-gfm/unnamed-chunk-13-1.png)<!-- -->

``` r
bic_error <- rep(NA, 20)
for(i in 1:20){
  yhat_train <- nadaraya_watson(mcycle_train$accel, x_train, x_train,kernel_k_nearest_neighbors, knn=i)
  edf <- effective_df(y, x, kernel_k_nearest_neighbors, knn=i)
  bic_error[i] <- bic(y, yhat_train, edf)
}
print(bic_error)
```

    ##  [1] 366.9173 429.1488 503.3306 490.3909 503.6229 544.0923 556.1511 544.2790
    ##  [9] 561.0320 584.8314 582.0772 587.1785 602.1218 609.6593 617.4826 637.6202
    ## [17] 658.5265 671.2539 685.3829 718.9715

``` r
plot(seq(1:20), bic_error, xlab = 'k',ylab = 'BIC')
```

![](Homework_4_files/figure-gfm/unnamed-chunk-14-1.png)<!-- -->

``` r
valid_error <- rep(NA, 20)
for(i in 1:20){
  yhat_valid <- nadaraya_watson(mcycle_test$accel, x_valid, x_valid,kernel_k_nearest_neighbors, knn=i)
  valid_error[i] <-  error(mcycle_test$accel,yhat_valid)
}

#Error
#training error
plot(seq(1:20), valid_error, xlab = 'k', ylab = 'testing error')
```

![](Homework_4_files/figure-gfm/unnamed-chunk-15-1.png)<!-- -->

``` r
## create a grid of inputs 
x_plot <- matrix(seq(min(x),max(x),length.out=100),100,1)

## make predictions using NW method at each of grid points
y_hat_plot <- nadaraya_watson(y, x, x_plot,
  kernel_epanechnikov, lambda=1)

## plot predictions
plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
lines(x_plot, y_hat_plot, col="#882255", lwd=2) 
```

![](Homework_4_files/figure-gfm/unnamed-chunk-16-1.png)<!-- -->

``` r
# # how does k affect shape of predictor and eff. df using k-nn kernel ?
# manipulate({
#   ## make predictions using NW method at training inputs
#   y_hat <- nadaraya_watson(y, x, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   edf <- effective_df(y, x,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   aic_ <- aic(y, y_hat, edf)
#   bic_ <- bic(y, y_hat, edf)
#   y_hat_plot <- nadaraya_watson(y, x, x_plot,
#     kern=kernel_k_nearest_neighbors, k=k_slider)
#   plot(x, y, xlab="Time (ms)", ylab="Acceleration (g)")
#   legend('topright', legend = c(
#     paste0('eff. df = ', round(edf,1)),
#     paste0('aic = ', round(aic_, 1)),
#     paste0('bic = ', round(bic_, 1))),
#     bty='n')
#   lines(x_plot, y_hat_plot, col="#882255", lwd=2)
# }, k_slider=slider(1, 10, initial=3, step=1))
```

## For each value of the tuning parameter, Perform 5-fold cross-validation using the combined training and validation data. This results in 5 estimates of test error per tuning parameter value.

``` r
## 5-fold cross-validation of knnreg model
## create five folds
set.seed(1985)
mcycle_flds  <- createFolds(mcycle$times, k=5)
print(mcycle_flds)
```

    ## $Fold1
    ##  [1]   4   7  10  12  13  22  26  40  44  53  54  57  60  69  73  76  80  84  93
    ## [20] 100 118 120 122 129 130 132
    ## 
    ## $Fold2
    ##  [1]   3  15  16  21  28  32  34  43  47  48  49  51  55  59  83  85  87  89  92
    ## [20]  96  98 104 112 116 117 125 131
    ## 
    ## $Fold3
    ##  [1]   2  11  14  19  25  31  36  42  46  50  52  66  67  75  77  79  88  91  97
    ## [20] 101 105 109 111 121 126 127
    ## 
    ## $Fold4
    ##  [1]   6   8   9  17  29  30  33  35  38  39  45  64  65  70  71  74  81  82  94
    ## [20]  99 102 103 106 108 114 124 133
    ## 
    ## $Fold5
    ##  [1]   1   5  18  20  23  24  27  37  41  56  58  61  62  63  68  72  78  86  90
    ## [20]  95 107 110 113 115 119 123 128

``` r
sapply(mcycle_flds, length)  ## not all the same length
```

    ## Fold1 Fold2 Fold3 Fold4 Fold5 
    ##    26    27    26    27    27

``` r
cvknnreg <- function(kNN = 10, flds=mcycle_flds) {
  cverr <- rep(NA, length(flds))
  for(tst_idx in 1:length(flds)) { ## for each fold
    
    ## get training and testing data
    mcycle_trn <- mcycle[-flds[[tst_idx]],]
    mcycle_tst <- mcycle[ flds[[tst_idx]],]
    
    ## fit kNN model to training data
    knn_fit <- knnreg(accel ~ times,
                      k=kNN, data=mcycle_trn)
    
    ## compute test error on testing data
    pre_tst <- predict(knn_fit, mcycle_tst)
    cverr[tst_idx] <- mean((mcycle_tst$accel - pre_tst)^2)
  }
  return(cverr)
}

## Compute 5-fold CV for kNN = 1:20
cverrs <- sapply(1:20, cvknnreg)
print(cverrs) ## rows are k-folds (1:5), cols are kNN (1:20)
```

    ##           [,1]      [,2]     [,3]     [,4]     [,5]     [,6]     [,7]     [,8]
    ## [1,]  873.7206  700.0270 689.3987 485.3409 434.3789 395.7523 363.1052 359.1610
    ## [2,] 1270.1977  875.9311 663.4668 680.9788 726.2984 668.5983 697.4114 690.2885
    ## [3,] 1372.8886  716.4393 679.7326 687.8899 673.3354 636.9632 651.5842 632.1199
    ## [4,]  905.4988 1056.9299 917.4261 791.3737 838.6811 795.8138 807.7914 833.1298
    ## [5,]  832.2273  625.6535 485.9449 488.5612 507.9257 469.1820 487.2257 489.8112
    ##          [,9]    [,10]    [,11]    [,12]    [,13]    [,14]    [,15]    [,16]
    ## [1,] 368.4566 375.3212 369.9732 352.7022 364.5059 386.6813 382.1285 372.2810
    ## [2,] 727.3307 739.7056 718.3590 725.5623 720.8505 762.2420 778.9767 720.5379
    ## [3,] 647.5598 657.6890 734.9424 688.5695 682.5284 684.5491 729.4468 753.2387
    ## [4,] 797.8092 770.4313 815.1868 794.5416 789.8681 728.4384 717.4297 745.2519
    ## [5,] 498.3470 531.5351 500.5685 542.4821 532.3482 497.1585 529.9872 586.2051
    ##         [,17]    [,18]    [,19]    [,20]
    ## [1,] 354.1053 347.2541 384.5101 394.2038
    ## [2,] 740.5859 799.7265 870.8813 912.4776
    ## [3,] 789.3431 759.4425 753.5538 778.6373
    ## [4,] 737.8144 734.2257 743.0353 752.7634
    ## [5,] 588.0262 601.9674 602.1801 618.6925

``` r
cverrs_mean <- apply(cverrs, 2, mean)
cverrs_sd   <- apply(cverrs, 2, sd)
```

## Plot the CV-estimated test error (average of the five estimates from each fold) as a function of the tuning parameter. Add vertical line segments to the figure (using the segments function in R) that represent one “standard error” of the CV-estimated test error (standard deviation of the five estimates from each fold).

``` r
## Plot the results of 5-fold CV for kNN = 1:20
plot(x=1:20, y=cverrs_mean, 
     ylim=range(cverrs),
     xlab="'k' in kNN", ylab="CV Estimate of Test Error")
segments(x0=1:20, x1=1:20,
         y0=cverrs_mean-cverrs_sd,
         y1=cverrs_mean+cverrs_sd)
best_idx <- which.min(cverrs_mean)
points(x=best_idx, y=cverrs_mean[best_idx], pch=20)
abline(h=cverrs_mean[best_idx] + cverrs_sd[best_idx], lty=3)
```

![](Homework_4_files/figure-gfm/unnamed-chunk-19-1.png)<!-- -->

## Interpret the resulting figures and select a suitable value for the tuning parameter.

\*\*As we can see in the plot above, when k = 6, the model has the
lowest test error. However, if k = 6, the model is kind of complex.
Since the test errors are close when k = 6 to k = 15. I may take a
middle one say that the suitable value is k = 10.
