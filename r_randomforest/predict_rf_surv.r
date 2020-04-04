suppressMessages(library(randomForestSRC)) # survival forest
suppressMessages(library(dplyr))
suppressMessages(library(progress))

Xtest <- read.csv('../data/interim/aids_x_test.csv')
ytest <- read.csv('../data/interim/aids_y_test.csv')

testing <- cbind(Xtest, ytest)

model <- readRDS("../models/01_best_rfsurv.rds")

pred <- predict(model, testing); pred

writeLines(paste0("C-index: ", 1-na.omit(pred$err.rate)))

bootstrap_cindex <- function(mod, dat, N=1000) {
    pb <- progress_bar$new(total = N)
	c.indx <- NULL
    n <- nrow(dat)

	for (i in 1:N) {
        pb$tick()
        use <- sample(1:n, n, replace=TRUE)
        new_dat <- dat[use, ]
        score <- predict(mod, new_dat)$err.rate %>% na.omit
        c.indx <- c(c.indx, 1 - score)
	}
    return(c.indx)
}

scores <- bootstrap_cindex(model, testing) 
avg.c <- mean(scores)
st.err.c <- sd(scores)
lower.c <- qnorm(.025, avg.c, st.err.c) %>% round(3)
upper.c <- qnorm(.975, avg.c, st.err.c) %>% round(3)

writeLines(paste0("Bootstrapped C-index: ", round(avg.c, 3)))
writeLines(paste0("95% CI: ", lower.c, " - ", upper.c))
