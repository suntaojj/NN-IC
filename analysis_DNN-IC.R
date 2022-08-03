########### Neural network survival prediction for interval-censored data #############

library(keras)
library(tensorflow)
library(survival)

# parameter setup
epoch <- 1000
batch_size <- 50
num_nodes <- 50
string_activation <- "selu"
num_l1 <- 0.1
num_dropout <- 0
num_lr <- 0.0002
num_layer <- 2
num_m = 3

# load codes and data
source("fun_DNN-IC.R")
load("interval_data.RData")
dat = interval_data[,c("Left","Right","status",paste0("SNP_", 1:20))]
left_dat = as.matrix(dat[, c("Left")], nrow = nrow(dat))
right_dat = as.matrix(dat[, c("Right")], nrow = nrow(dat))
pred_dat = as.matrix(dat[, c(paste0("SNP_", 1:20))], nrow = nrow(dat))
status_dat = as.matrix(dat[, c("status")], nrow = nrow(dat))
right_dat[status_dat==0] = max(dat$Right[is.finite(dat$Right)]) + 0.1
num_l <- min(left_dat)
num_u <- max(right_dat)

### run DNN-IC ###
rm(model)
model <- build_model_ic(left_dat, right_dat, pred_dat, m = num_m, l = num_l, u = num_u, num_nodes, string_activation, num_l1, num_dropout, num_lr, num_layer)

model %>% fit(
  list(left_dat, right_dat, pred_dat),
  status_dat,
  epochs = epoch,
  batch_size = batch_size,
  verbose = 1
)

### Brier score ###
brier <-  brier_ic_nn(obj = model,
                      left_new = left_dat,
                      right_new = right_dat,
                      x_new = pred_dat,
                      type = "IBS")
brier


### AUC ###
lp <- model %>% predict(list(left_dat, right_dat, pred_dat))
lp = lp[,3]
Marker = (lp - min(lp))/(max(lp) - min(lp)) + 0.1 # need to be positive
U <- as.vector(left_dat)
V <- as.vector(right_dat)
Delta <- 2 # 1 for LC, 2 for IC, 3 for RC
U <- ifelse(dat$Left == 0, V, U) # for left-censor
Delta <- ifelse(dat$Left == 0, 1, 2) # for left-censor
V <- ifelse(dat$Right == Inf, U, V) # for right-censor
Delta <- ifelse(dat$Right == Inf, 3, Delta) # for left-censor

library(intcensROC)
res <- intcensROC(U, V, Marker, Delta, median(U), gridNumber = 500)
intcensAUC(res)


##### lime ######
library("lime")
assignInNamespace("model_type.keras.engine.training.Model",model_type.keras.engine.training.Model,ns="lime")
assignInNamespace("predict_model.keras.engine.training.Model", predict_model.keras.engine.training.Model,ns="lime")


explainer <- lime(
  x              = data.frame(pred_dat),
  model          = model,
  bin_continuous = F)

explanation <- lime::explain(
  x = data.frame(pred_dat),
  explainer    = explainer,
  n_features   = 10,
  feature_select = "auto",
  n_permutations = 1000)

plot_explanations(explanation)

