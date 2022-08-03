loss_ic <- function(y_true, y_pred) {

  output_left = y_pred[,1]
  output_right = y_pred[,2]
  output_pred = y_pred[,3]

  status = y_true

  gLl <- k_exp(output_pred) * output_left
  gLr <- k_exp(output_pred) * output_right
  status <- k_reshape(status, k_shape(gLl))
  status <- tf$cast(status,  tf$float32)
  lik_ic = k_exp(-1 * gLl) -  status * k_exp(-1 * gLr)

  lik_ic_nozeros = tf$boolean_mask(tensor = lik_ic, mask = tf$greater(lik_ic, 0))
  loglik_ic = k_sum(k_log(lik_ic_nozeros))

  return(tf$negative(loglik_ic))

}

bern <- function(j,m,l,u,t){
  j_new = tf$cast(j, dtype = tf$float32)
  m_new = tf$cast(m, dtype = tf$float32)
  BS = (tf$exp(tf$math$lgamma( k_cast(m_new + k_ones_like(m_new), dtype = tf$float32) ))/(tf$exp(tf$math$lgamma( k_cast(j_new + k_ones_like(m_new), dtype = tf$float32) ))*tf$exp(tf$math$lgamma(  k_cast(m_new - j_new + k_ones_like(m_new), dtype = tf$float32) ))))*(k_pow((t-l)/(u-l), j_new))*(k_pow( k_ones_like(m_new) -(t-l)/(u-l), m_new-j_new))
  return(BS)
}

SemiParLayer <- R6::R6Class("SemiParLayer",

                            inherit = KerasLayer,

                            public = list(

                              output_dim = NULL,
                              phi = NULL,
                              m = NULL,
                              l = NULL,
                              u = NULL,
                              m_new = NULL,

                              initialize = function(output_dim, m, l, u) {
                                self$output_dim <- output_dim
                                self$m <- m
                                self$l <- l
                                self$u <- u
                                self$m_new <- k_cast(m + 1, dtype = tf$int32)
                              },

                              build = function(input_shape) {

                                self$phi <- self$add_weight(
                                  name = 'phi',
                                  shape = list(input_shape[[2]], self$m_new),

                                  initializer = initializer_random_uniform(),
                                  trainable = TRUE
                                )

                              },

                              call = function(time, mask = NULL) {

                                b = tf$TensorArray(dtype = tf$float32, infer_shape = F, dynamic_size=T, size=self$m_new)

                                i = tf$cast(0, dtype = tf$int32)
                                loop_cond = function(i, ...) {return(i < tf$cast(self$m + 1, dtype = tf$int32))}

                                loop_body = function(i, m, l, u, time, b) {

                                  b_tmp <- bern(i, m, l, u, time)
                                  b = b$write(i, b_tmp)

                                  i_new = i + k_ones_like(i)

                                  return(list(i_new, m, l, u, time, b))
                                }

                                loop_out = tf$while_loop(cond = loop_cond, body = loop_body,
                                                         loop_vars = list(i, self$m, self$l, self$u, time, b)
                                )

                                bp = loop_out[length(loop_out)][[1]]

                                bp = bp$stack()
                                bp = k_transpose(k_batch_flatten(bp))

                                ep <- k_cumsum(k_exp(self$phi), axis = 0)
                                gL <- k_dot(bp, k_transpose(ep))
                                gL

                              },

                              compute_output_shape = function(input_shape) {
                                list(input_shape[[1]], self$output_dim)
                              }

                            )
)

layer_semipar <- function(object, output_dim, m, l, u, name = NULL, trainable = TRUE) {
  create_layer(layer_class = SemiParLayer,
               object = object,
               args = list(
                 output_dim = as.integer(output_dim),
                 m = as.integer(m),
                 l = l,
                 u = u,
                 name = name,
                 trainable = trainable
               )
  )}

ic_semipar <- function(y_true, y_pred) {

  output_left = y_pred[,1]
  output_right = y_pred[,2]
  output_pred = y_pred[,3]

  status = y_true

  gLl <- k_exp(output_pred) * output_left
  gLr <- k_exp(output_pred) * output_right

  lik_ic = k_exp(-1 * gLl) - status * k_exp(-1 * gLr)

  lik_ic_nozeros = tf$boolean_mask(tensor = lik_ic, mask = tf$greater(lik_ic, 0))
  loglik_ic = k_sum(k_log(lik_ic_nozeros))

  return(tf$negative(loglik_ic))

}

build_model_ic <- function(left, right, pred, m, l = 0, u = NULL, num_nodes, string_activation, num_l1, num_dropout, num_lr, num_layer) {



  ###### pred layer ######
  input_pred <- layer_input(shape = list(ncol(pred)),
                            dtype = "float32",
                            name = "pred")

  if (num_layer == 1) {
    output_pred <- input_pred %>%
      layer_dense(units = num_nodes, activation = string_activation, kernel_regularizer = regularizer_l1(num_l1)) %>%
      layer_dropout(rate = num_dropout) %>%
      layer_dense(units = 1, activation = NULL)
  } else if (num_layer == 2) {
    output_pred <- input_pred %>%
      layer_dense(units = num_nodes, activation = string_activation, kernel_regularizer = regularizer_l1(num_l1)) %>%
      layer_dropout(rate = num_dropout) %>%
      layer_dense(units = num_nodes, activation = string_activation, kernel_regularizer = regularizer_l1(num_l1)) %>%
      layer_dropout(rate = num_dropout) %>%
      layer_dense(units = 1, activation = NULL)
  }


  ###### time layers  ######
  input_left <- layer_input(shape = list(ncol(left)),
                            dtype = "float32",
                            name = "left")
  input_right <- layer_input(shape = list(ncol(right)),
                             dtype = "float32",
                             name = "right")
  encoded_time <- layer_semipar(output_dim = 1, m = m, l = l, u = u)
  output_left <- input_left %>% encoded_time
  output_right <- input_right %>% encoded_time
  concatenated <- layer_concatenate(list(output_left,
                                         output_right,
                                         output_pred),
                                    axis = -1)
  model <- keras_model(inputs = list(input_left, input_right, input_pred),
                       outputs = concatenated)
  model %>% compile(
    optimizer = optimizer_rmsprop(lr = num_lr),
    loss = loss_ic,
    metrics = NULL
  )

}

brier_ic_nn <- function(obj, left_new, right_new, x_new, btime = NULL, type = c("IBS", "BS")) {

  N <- nrow(left_new)
  intL <- left_new
  intR <- right_new
  times <- c(intL, intR)
  if (is.null(btime)) btime=range(intL, intR)

    out.1 <- obj %>% predict(list(left_new, right_new, x_new))
    out.2 <- obj %>% predict(list(times, times, rbind(x_new, x_new)))
    S_times = t(k_get_value(exp(-1 * (k_dot(k_cast(matrix(exp(out.1[,3]), ncol = 1), dtype = tf$float32), k_cast(matrix(out.2[,1], nrow = 1), dtype = tf$float32))))))


    if (length(btime) < 1)
      stop("btime not given")
    if (length(btime) != 2) {
      stop("btime should be a vector of two indicating the range of the times")
    }
    else {
      if (btime[1] < min(times))
        warning("btime[1] is smaller than min(times)")
      if (btime[2] > max(times))
        warning("btime[2] is larger than max(times)")
      btime <- unique(times[times >= btime[1] & times <= btime[2]])
      btime <- sort(btime)
    }


    bsc <- rep(0, length(times))
    for (i in 1:length(times)) {
      bsc_temp <- matrix(0, nrow = N, ncol = 1)
      k = 0
      for (j in 1: N) {
        St <- S_times[i, j]
        if (times[i] <= intL[j]) {
          IY = 1
        }
        else if (times[i] > intR[j]) {
          IY = 0
        }
        else {
          Sr <- S_times[N + j, j]
          Sl <- S_times[j, j]
          if (Sl != Sr) {
            IY = (St - Sr)/(Sl - Sr)
          }
          else {
            IY = NULL
          }
        }
        if (is.null(IY) == 0) {
          k = k + 1
          bsc_temp[k] = (IY - St)^2
        }
      }
      bsc[i] = mean(bsc_temp[1:k])
    }

    ot = order(times)
    bsc = bsc[ot]
    times = times[ot]
    unik <- !duplicated(times)
    bsc <- bsc[unik]
    times <- times[unik]
    bsc <- bsc[times <= max(btime)]
    if (type == "IBS") {
      idx <- 2:length(btime)
      RET <- diff(btime) %*% ((bsc[idx - 1] + bsc[idx])/2)
      if (max(btime)/tail(btime, n = 2)[1] > 1000 && tail(bsc,
                                                          n = 1) == 0) {
        RET <- RET - (tail(btime, n = 1) - tail(btime, n = 2)[1]) *
          ((tail(bsc, n = 1) + tail(bsc, n = 2)[1])/2)
        btime <- head(btime, n = -1)
      }

      RET <- RET/diff(range(btime))
      names(RET) <- "integrated Brier score"
      attr(RET, "time") <- range(btime)
    } else if (type == "BS") {
      RET <- bsc
      names(RET) <- btime
      attr(RET, "type") <- "Brier score"
    } else {
      stop("unknown type of results")
    }
    RET

}


model_type.keras.engine.training.Model <- function(x, ...) {
  return("regression")
}

predict_model.keras.engine.training.Model <- function(x, newdata, type = "raw", ...) {
  newdata <- list(as.matrix(newdata[,1]), as.matrix(newdata[,1])+1, as.matrix(newdata))
  y_dat_pred <- x %>% predict(newdata)
  y_dat_pred <- y_dat_pred[,3]
  return(data.frame(Response = y_dat_pred))
}

