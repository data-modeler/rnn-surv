library(randomForestSRC) # survival forest
library(caret) # useful machine learning functions
library(rBayesianOptimization) # for hyperparam tuning

Xtrain <- read.csv('../../data/interim/aids_x_train.csv')
ytrain <- read.csv('../../data/interim/aids_y_train.csv')

training <- cbind(Xtrain, ytrain)

surv_f = as.formula(Surv(tte,event) ~ .)

# Number of trees in random forest
trees <- as.integer(c(200, 2000))

# Number of features to consider at every split
mtry <- as.integer(c(2, 4)) # default is sqrt(p)

# Maximum number of levels in tree
nodedepth <- NULL  # as.integer(c(10000, (1:11)*10))

# Minimum number of samples required at each leaf node
nodesize <- as.integer(c(1, 10))

# Method of selecting samples for training each tree
bootstrap <- 'by.root' # c('by.root', 'by.node') # has to be by.root since we're using OOB for cv


rfsc_opt <- function(t, m, ns) {
  
  model <- rfsrc(formula=surv_f, data=training, 
		 ntree=t, 
		 bootstrap='by.root',
		 mtry=m, 
		 nodesize=ns,
		 nodedepth=NULL,
		 importance = 'permute')
  return(
    list(
      Score = 1 - model$err.rate[t], # C-index to be maximized
      Pred = model$predicted.oob # the validation/cross-validation prediction for ensembling/stacking
    )
  )  
}


OPT_Res <- BayesianOptimization(rfsc_opt,
				bounds = list(t = trees,
					      m = mtry,
					      ns = nodesize
					      ),
				init_grid_dt = NULL, init_points = 10, n_iter = 20,
				acq = "ucb", kappa = 2.576, eps = 0.0,
				verbose = TRUE)

best_model <- rfsrc(formula=surv_f, data=training, 
		    ntree=OPT_Res$Best_Par[1], 
		    bootstrap='by.root',
		    mtry=OPT_Res$Best_Par[2], 
		    nodesize=OPT_Res$Best_Par[3],
		    nodedepth=NULL,
		    importance = 'permute')

saveRDS(best_model, "../../models/01_best_rfsurv.rds")

