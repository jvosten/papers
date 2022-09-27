
# Each model follows the same blueprint:
# - parameter specification (image shape, # of epochs, etc.)
# - data set generation
# - model specification
# - run & evaluate model
# - save results

# Baseline Model *************************************************************************************************************** #

sample_path("tiny")
image_shape <- c(150, 150, 3)
target_size <- image_shape[-3]
batch_size <- 32
epochs <- 50
output_n <- length(classes_update())
base_model_spec <- list(`model name` = "Base CNN", classes = output_n, 
                       `# epochs` = epochs, `batch size` = batch_size, 
                       `image shape` = paste(image_shape, collapse = ", "), optimizer = "RMS prop")

base_train <- data_generator(train_dir)
base_valid <- data_generator(valid_dir)
base_test <- data_generator(test_dir)

# test_pic(base_train)
# str(reticulate::iter_next(base_train))

# Model ------------------------------------------------------------------------------------------------------------------------ # 

base_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = image_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = output_n, activation = "sigmoid")

base_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), 
  metrics = c("acc")
)
# summary(base_model)

base_rt <- run_time(
base_history <- base_model %>% fit_generator( 
  base_train,
  steps_per_epoch = base_train$n / base_train$batch_size,
  epochs = epochs, 
  validation_data = base_valid,
  validation_steps = 2
))

base_model %>% save_model(., "base_model")

base_eval <- base_model %>% 
  evaluate_generator(base_test, steps = 5)

base_res <- list(`model name` = "Base CNN", classes = output_n, 
                 `training acc` = base_history$metrics$acc[epochs], 
                 `validation acc` = base_history$metrics$val_acc[epochs], 
                 `test accuracy` = base_eval[[2]], `run time` = base_rt$run_time[2])

# Baseline Model 25 classes **************************************************************************************************** #

sample_path("small")
image_shape <- c(150, 150, 3)
target_size <- image_shape[-3]
batch_size <- 32
epochs <- 50
output_n <- length(classes_update())
base25_model_spec <- list(`model name` = "Base CNN", classes = output_n, 
                          `# epochs` = epochs, `batch size` = batch_size, 
                          `image shape` = paste(image_shape, collapse = ", "), optimizer = "RMS prop")

base_train <- data_generator(train_dir)
base_valid <- data_generator(valid_dir)
base_test <- data_generator(test_dir)

# test_pic(base_train)
# str(reticulate::iter_next(base_train))

# Model ------------------------------------------------------------------------------------------------------------------------ # 

base25_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = image_shape) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dense(units = 256, activation = "relu") %>% 
  layer_dense(units = output_n, activation = "sigmoid")

base25_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), 
  metrics = c("acc")
)
# summary(base25_model)

base25_rt <- run_time(
  base25_history <- base25_model %>% fit_generator( 
    base_train,
    steps_per_epoch = base_train$n / base_train$batch_size,
    epochs = epochs, 
    validation_data = base_valid,
    validation_steps = 4
  ))

base25_model %>% save_model(., "base25_model")

base25_eval <- base25_model %>% 
  evaluate_generator(base_test, steps = 50)

base25_res <- list(`model name` = "Base CNN", classes = output_n, 
                   `training acc` = base25_history$metrics$acc[epochs], 
                   `validation acc` = base25_history$metrics$val_acc[epochs], 
                   `test accuracy` = base25_eval[[2]], `run time` = base25_rt$run_time[2])

# Augmented Model ************************************************************************************************************** #

sample_path("tiny")
image_shape <- c(150, 150, 3)
target_size <- image_shape[-3]
batch_size <- 32
epochs <- 50
output_n <- length(classes_update())
aug_model_spec <- list(`model name` = "Augmented CNN", classes = output_n, 
                       `# epochs` = epochs, `batch size` = batch_size, 
                       `image shape` = paste(image_shape, collapse = ", "), optimizer = "RMS prop")

# instead of only rescaling the pictures they get transformed by a variety of other operations
aug_datagen <- image_data_generator( 
  rescale = 1/255, 
  rotation_range = 40, 
  width_shift_range = 0.2, 
  height_shift_range = 0.2, 
  shear_range = 0.2,
  zoom_range = 0.2, 
  horizontal_flip = TRUE, 
  fill_mode = "nearest"
)

aug_train <- data_generator(directory = train_dir, generator = aug_datagen, target_size)
aug_valid <- data_generator(valid_dir)
aug_test <- data_generator(test_dir)

# test_pic(aug_train)
# str(reticulate::iter_next(aug_train))

# Model ------------------------------------------------------------------------------------------------------------------------ # 

aug_model <- keras_model_sequential() %>%
  layer_conv_2d(filters = 32, kernel_size = c(3, 3), activation = "relu",
                input_shape = c(150, 150, 3)) %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>% 
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_flatten() %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 512, activation = "relu") %>% 
  layer_dense(units = output_n, activation = "sigmoid")

aug_model %>% compile(
  loss = "categorical_crossentropy",
  optimizer = optimizer_rmsprop(lr = 1e-4), 
  metrics = c("acc")
)

aug_rt <- run_time(
aug_history <- aug_model %>% fit_generator( 
  aug_train, 
  steps_per_epoch = 100,
  epochs = epochs,
  validation_data = aug_valid,
  validation_steps = 20
))
# summary(aug_model)

aug_model %>% save_model(., "aug_model")

aug_eval <- aug_model %>% 
  evaluate_generator(aug_test, steps = 50)

aug_res <- list(`model name` = "Augmented CNN", classes = output_n,
                `training acc` = aug_history$metrics$acc[50], 
                `validation acc` = aug_history$metrics$val_acc[50], 
                `test accuracy` = aug_eval[[2]], `run time` = aug_rt$run_time[2])

# Pretrained Model I ********************************************************************************************************** #

sample_path("small")
image_shape <- c(150, 150, 3)
target_size <- image_shape[-3]
batch_size <- 25
epochs <- 50
output_n <- length(classes_update())
vgg_model_spec <- list(`model name` = "VGG CNN", classes = output_n, 
                       `# epochs` = epochs, `batch size` = batch_size, 
                       `image shape` = paste(image_shape, collapse = ", "), optimizer = "RMS prop")

# setting up the convolutional base
conv_base <- application_vgg16(
  weights = "imagenet",
  include_top = FALSE,
  input_shape = image_shape
)

# wrapper function to use the vgg16 convolutional base for feature extraction
extract_features <- function(directory, sample_count) {
  
  features <- array(0, dim = c(sample_count, 4, 4, 512))
  labels <- array(0, dim = c(sample_count, length(classes_update())))
  
  # image get loaded in batches
  generator <- flow_images_from_directory(
    directory = directory,
    generator = datagen,
    target_size = target_size,
    batch_size = batch_size,
    class_mode = "categorical"
  )
  
  # for every batch until a chosen amount of sample img is reached, the conv_base is 
  # applied on every single picture for feature extraction using the predict function
  i <- 0
  while(TRUE) {
    batch <- generator_next(generator)
    inputs_batch <- batch[[1]]
    labels_batch <- batch[[2]]
    features_batch <- conv_base %>% predict(inputs_batch)
    
    index_range <- ((i * batch_size)+1):((i + 1) * batch_size)
    features[index_range,,,] <- features_batch
    labels[index_range,]  <- labels_batch
    
    i <- i + 1
    if (i * batch_size >= sample_count)
      break
  }
  list(
    features = features,
    labels = labels
  )
}

vgg_train <- extract_features(train_dir, 2400)
vgg_valid <- extract_features(valid_dir, 125)
vgg_test <- extract_features(test_dir, 125)

# Check data sets
nrow(vgg_train$labels)
nrow(vgg_test$labels)
nrow(vgg_valid$labels)

# Reshape features to preapre them for the keras model
reshape_features <- function(features) {
  array_reshape(features, dim = c(nrow(features), 4 * 4 * 512))
}
vgg_train$features <- reshape_features(vgg_train$features)
vgg_valid$features <- reshape_features(vgg_valid$features)
vgg_test$features <- reshape_features(vgg_test$features)

# Model ------------------------------------------------------------------------------------------------------------------------ # 

vgg_model <- keras_model_sequential() %>%
  layer_dense(units = 256, activation = "relu",
              input_shape = 4 * 4 * 512) %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = output_n, activation = "sigmoid")

vgg_model %>% compile(
  optimizer = optimizer_rmsprop(lr = 1e-4),
  loss = "categorical_crossentropy",
  metrics = c("accuracy")
)
# summary(vgg_model)

vgg_rt <- run_time(
vgg_history <- vgg_model %>% fit(
  vgg_train$features, vgg_train$labels,
  epochs = epochs,
  steps_per_epoch = nrow(vgg_train$features) / batch_size,
  batch_size = batch_size,
  validation_data = list(vgg_valid$features, vgg_valid$labels),
  validation_steps = nrow(vgg_valid$features) / batch_size
))

vgg_model %>% save_model(., "vgg_model")

vgg_eval <- vgg_model %>%
  evaluate(x = vgg_test$features, y = vgg_test$labels, batch_size = batch_size)

vgg_res <- list(`model name` = "VGG CNN", classes = output_n,
                `training acc` = vgg_history$metrics$acc[epochs], 
                `validation acc` = vgg_history$metrics$val_acc[epochs], 
                `test accuracy` = vgg_eval[[2]], `run time` = vgg_rt$run_time[2])

# Pretrained Model II ********************************************************************************************************* #

sample_path("small")
image_shape <- c(224L, 224L, 3L)
target_size <- image_shape[-3]
batch_size <- 32
epochs <- 25
output_n <- length(classes_update())
mv2_model_spec <- list(`model name` = "MV2 CNN", classes = output_n, 
                       `# epochs` = epochs, `batch size` = batch_size, 
                       `image shape` = paste(image_shape, collapse = ", "), optimizer = "Adam")

mv2_train <- data_generator(train_dir, generator = datagen,  target_size = target_size)
mv2_valid <- data_generator(valid_dir, generator = datagen, target_size = target_size)
mv2_test <- data_generator(test_dir, generator = datagen, target_size = target_size)

# test_pic(mv2_train) 
# str(reticulate::iter_next(mv2_train))

# Model ------------------------------------------------------------------------------------------------------------------------ # 

# creating a feature extractor layer as a step of the actual model; input data does not need any additional preprocessing
feature_extractor_url <- "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer <- layer_hub(handle = feature_extractor_url, 
                                     input_shape = image_shape)

freeze_weights(feature_extractor_layer)

mv2_model <- keras_model_sequential(list(
  feature_extractor_layer,
  layer_dense(units = output_n, activation='softmax')
))

mv2_model %>% compile(
  optimizer = optimizer_adam(lr = 1e-4), 
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)
# summary(mv2_model)

mv2_rt <- run_time(
mv2_history <- mv2_model %>% fit_generator(
  mv2_train, 
  epochs= epochs, 
  steps_per_epoch = mv2_train$n / mv2_train$batch_size,
  verbose = TRUE,
  validation_data = mv2_valid, 
  validation_steps = mv2_valid$n /mv2_valid$batch_size 
))

mv2_model %>% save_model(., "mv2_model")

mv2_eval <- mv2_model %>% 
  evaluate_generator(mv2_test, steps = 50)

mv2_res <- list(`model name` = "MV2 CNN", classes = output_n, 
                `training acc` = mv2_history$metrics$accuracy[epochs], 
                `validation acc` = mv2_history$metrics$val_accuracy[epochs], 
                `test accuracy` = mv2_eval[[2]], `run time` = mv2_rt$run_time[2])

image_batch <- reticulate::iter_next(mv2_test)
predictions <- predict_classes(mv2_model, image_batch[[1]])


colf <- function() color <- '#bb0000'
par(mfcol = c(4,8), mar = rep(1, 4), oma = rep(0.2, 4))
image_batch[[1]] %>% 
  purrr::array_tree(1) %>%
  purrr::set_names(names(mv2_train$class_indices)[predictions + 1]) %>% 
  purrr::map(as.raster) %>%
  purrr::iwalk(~{plot(.x); title(.y, col.main = colf())})

# Pretrained Model II (210 Classes) ******************************************************************************************** #

sample_path("orig")
image_shape <- c(224L, 224L, 3L)
target_size <- image_shape[-3]
batch_size <- 32
epochs <- 10
output_n <- length(classes_update())
mv2_210_model_spec <- list(`model name` = "MV2 CNN", classes = output_n, 
                       `# epochs` = epochs, `batch size` = batch_size, 
                       `image shape` = paste(image_shape, collapse = ", "), optimizer = "Adam")

mv2_train <- data_generator(train_dir, generator = datagen,  target_size = target_size)
mv2_valid <- data_generator(valid_dir, generator = datagen, target_size = target_size)
mv2_test <- data_generator(test_dir, generator = datagen, target_size = target_size)

# test_pic(mv2_train) 
# str(reticulate::iter_next(mv2_train))

# Model ------------------------------------------------------------------------------------------------------------------------ # 

feature_extractor_url <- "https://tfhub.dev/google/imagenet/mobilenet_v2_100_224/feature_vector/4"
feature_extractor_layer <- layer_hub(handle = feature_extractor_url, 
                                     input_shape = image_shape)


freeze_weights(feature_extractor_layer)

mv2_210_model <- keras_model_sequential(list(
  feature_extractor_layer,
  layer_dense(units = output_n, activation='softmax')
))

mv2_210_model %>% compile(
  optimizer = optimizer_adam(lr = 1e-4), 
  loss = "categorical_crossentropy",
  metrics = "accuracy"
)

# summary(mv2_210_model)
mv2_210_rt <- run_time(
  mv2_210_history <- mv2_210_model %>% fit_generator(
    mv2_train, 
    epochs = epochs, 
    steps_per_epoch = mv2_train$n / mv2_train$batch_size,
    verbose = TRUE,
    validation_data = mv2_valid, 
    validation_steps = mv2_valid$n /mv2_valid$batch_size 
))

mv2_210_model %>% save_model(., "mv2_210_model")

mv2_210_eval <- mv2_210_model %>% 
  evaluate_generator(mv2_test, steps = 50)

mv2_210_res <- list(`model name` = "MV2 CNN", classes = output_n, 
                `training acc` = mv2_210_history$metrics$accuracy[epochs], 
                `validation acc` = mv2_210_history$metrics$val_accuracy[epochs], 
                `test accuracy` = mv2_210_eval[[2]], `run time` = mv2_210_rt$run_time[2])

# Results ********************************************************************************************************************** #

res <- bind_rows(base_res, base25_res, aug_res, 
                 vgg_res, mv2_res, mv2_210_res)
models_spec <- bind_rows(base_model_spec, base25_model_spec, aug_model_spec, 
                         vgg_model_spec, mv2_model_spec, mv2_210_model_spec)
plots <- list("base_model" = base_history, "base_25_model" = base25_history, "aug_model" = aug_history, 
              "vgg_model" = vgg_history, "mv2_model" = mv2_history, "mv2_210_model" = mv2_210_history)

# save(class_hist,
#      res,
#      models_spec,
#      plots,
#      file = "./out/results.RData")
#
