# Set working directory
setwd("~/GitHub/pgdtw/sDTW")

dataComplete = c(1024, 1045, 1025, 1133, 1038, 1097, 1081, 968, 1045, 1081, 1016, 1055, 957, 1079, 1104, 1082, 1021, 1100, 1056, 1079, 1087, 1126, 1716, 1335, 982, 1106, 1066, 1127, 1306, 1159, 940, 1324)

batches = c()

for (filename in list.files(path = "data/prediction")){

    if (startsWith(filename, "pred")){
        batches = c(batches, substr(filename, 5, 8))
    }
}

update.hyperparameters = function(prior.hp, x){
    
    n = length(x)
    xbar = mean(x)
    sse = sum((x-xbar)^2)
    
    mu = ( prior.hp$nu0 * prior.hp$mu0 + n*xbar ) / ( prior.hp$nu0 + n )
    nu = prior.hp$nu0 + n
    alpha = prior.hp$alpha0 + n/2
    beta = prior.hp$beta0 + sse/2 + ( n * prior.hp$nu0 ) / ( prior.hp$nu0 + n ) * 0.5*( xbar - prior.hp$mu0 )^2
    
    posterior.hyperparameters = list(mu = mu,
                                     nu = nu,
                                     alpha = alpha,
                                     beta = beta)
}

qt_ls <- function(prob, df, mu, sp) qt(prob, df)*sp + mu

for (batchID in batches){

    df = read.csv(paste("data/prediction/pred", batchID, ".csv", sep = ""))
    
    true_length = df$T_query[1]
    
    i = 1
    for (len in dataComplete){
        if (len == true_length){
            break
        }
        i = i + 1
    }
    
    data = dataComplete[-i]
    
    conf = 0.99
    lowP = (1-conf)/2
    highP = conf+lowP
    
    prior.hyperparameters = list(mu0 = mean(data), 
                                 nu0 = length(data),
                                 alpha0 = length(data)/2,
                                 beta0 = sum((data- mean(data))^2)/2
    )
    
    intervals = data.frame(t = NA, low = NA, high = NA, lowDistro = NA)
    
    for (n in df$t_query) {
        x = df$predGB[seq(1, n)]
        posterior.hyperparameters = update.hyperparameters(prior.hyperparameters, x)
        
        sp = posterior.hyperparameters$beta*(posterior.hyperparameters$nu + 1)/(posterior.hyperparameters$nu * posterior.hyperparameters$alpha)
        
        new.row = data.frame(t = n,
                             low = max(c(n, qt_ls(prob = lowP, 
                                         df = 2*posterior.hyperparameters$alpha,
                                         mu = posterior.hyperparameters$mu,
                                         sp = sqrt(sp)))),
                             high = qt_ls(prob = highP, 
                                          df = 2*posterior.hyperparameters$alpha,
                                          mu = posterior.hyperparameters$mu,
                                          sp = sqrt(sp)),
                             lowDistro = qt_ls(prob = lowP, 
                                               df = 2*posterior.hyperparameters$alpha,
                                               mu = posterior.hyperparameters$mu,
                                               sp = sqrt(sp))) 
        intervals[n, ] = new.row
    }
    
    png(paste("data/prediction/plots/",batchID, ".png", sep = ""), width = 10, height = 8, units = "in", res = 300)
    
    plot(intervals$t, rep(true_length, n), type = "l", main = paste("Gradient Boosting, Mean Prediction", batchID), sub = paste(n,"/", true_length, " Minutes"), xlab = "t (min)", ylab = "Length (min)")
    
    polygon(c(intervals$t,rev(intervals$t)),c(intervals$lowDistro,rev(intervals$high)),col = "gray88", border = FALSE)
    polygon(c(intervals$t,rev(intervals$t)),c(intervals$low,rev(intervals$high)),col = "gray80", border = FALSE)
    
    
    lines(intervals$t, rep(true_length, n), lwd = 2)
    library(Hmisc)
    minor.tick(ny=5)
    grid()
    
    
    #add lines on borders of polygon
    lines(intervals$t, intervals$high, col="gray60",lty=2)
    lines(intervals$t, intervals$low, col="gray60",lty=2)
    lines(df$t_query, df$predGB, col = "red", lwd = 2)
    lines(df$t_query, df$predAbs, col = "green", lwd = 2)
    
    legend("bottomright", legend = c("True Value", "GB Prediction", "Absolute Estimate", paste(conf*100,"% Confidence Region on GB Prediction")), fill = c("black", "red", "green", "gray80"))
    dev.off()

}
