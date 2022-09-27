# Helper functions ************************************************************************************************************** #

# helper function to create directories ----------------------------------------------------------------------------------------- #

sub_dirs <- function(directory){
  train_dir <- file.path(directory, "train")
  dir.create(train_dir)
  valid_dir <- file.path(directory, "valid")
  dir.create(valid_dir)
  test_dir <- file.path(directory, "test")
  dir.create(test_dir)
}

# save_model wrapper ------------------------------------------------------------------------------------------------------------ #

save_model <- function(model, fname) {
  old <- setwd(here())
  on.exit(setwd(old), add = TRUE)
  
  setwd(here("output"))
  save_model_tf(model, filepath = fname)
}

# helper function to quickly change directories --------------------------------------------------------------------------------- #

sample_path <- function(size){
  if(size == "small"){
    base_dir <<- "./data/small_sample"
  } else if(size == "tiny"){
    base_dir <<- "./data/tiny_sample"
  } else {
    base_dir <<- "./data"
  }
  train_dir <<- file.path(base_dir, "train")
  valid_dir <<- file.path(base_dir, "valid")
  test_dir <<- file.path(base_dir, "test")
}

# sample function to sample images from the orig data set ----------------------------------------------------------------------- #

sampling <- function(nr_cl){
  classes_orig <- list.dirs(train_path, full.names = FALSE, recursive = FALSE)
  class_list <- sample(classes_orig, nr_cl)
  path_list <- c(test_path, train_path, valid_path)
  dir_list <- c(test_dir, train_dir, valid_dir)
  
  sampler <- function(path, dirc, class_new){
    for (i in seq_along(class_new)) {
      dir <- file.path(dirc, class_new[[i]])
      dir.create(dir)
    }
    new_dir <- list.dirs(dirc, full.names = TRUE, recursive = FALSE)
    
    map2(.x = sort(class_new), .y = new_dir, ~R.utils::copyDirectory(file.path(path, pattern = .x), file.path(.y)))
  }
  map2(.x = path_list, .y = dir_list, ~sampler(.x, .y, class_list))
  
  return(cat("Small sample data (", nr_cl, "classes) is done!", "\n"))
}

# Example augmented picture ----------------------------------------------------------------------------------------------------- # 

aug_example_pic <- function(){  
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
  
  fnames <- list.files(train_dir, full.names = TRUE, recursive = TRUE)
  img_path <- fnames[[sample(length(fnames), 1)]]
  
  img <- image_load(img_path, target_size = c(150, 150)) 
  img_array <- image_to_array(img)
  img_array <- array_reshape(img_array, c(1, 150, 150, 3))
  
  augmentation_generator <- flow_images_from_data( 
    img_array,
    generator = aug_datagen,
    batch_size = 1
  )
  
  
  op <- par(mfrow = c(2, 2), pty = "s", mar = c(1, 0, 1, 0)) 
  for (i in 1:4) {
    batch <- generator_next(augmentation_generator)
    plot(as.raster(batch[1,,,])) 
  }
  par(op)
}

# Example batch picture -------------------------------------------------------------------------------------------------------- #

# Takes a batch of pictures of a data set and prints them; helpful for checking if ds creation worked out
test_pic <- function(data){
  image_batch <- reticulate::iter_next(data)
  colnames(image_batch[[2]]) <- c(1:length(data$class_indices))
  true_labels <- as_tibble(image_batch[[2]])
  true_labels <- names(true_labels)[max.col(true_labels)]
  true_labels <- as.integer(true_labels)
  
  index <- 1:30
  op <- par(mfcol = c(5,6), mar = rep(1, 4), oma = rep(0.2, 4))
  image_batch[[1]][index,,,] %>% 
    purrr::array_tree(1) %>%
    purrr::set_names(names(data$class_indices[true_labels[index]])) %>% 
    purrr::map(as.raster) %>%
    purrr::iwalk(~{plot(.x); title(.y)})
  par(op)
}

# Save picture
# png(file="bird_ex.png", width = 1060, height = 530)
# test_pic(test_data)
# dev.off()

# Create a run time measure ---------------------------------------------------------------------------------------------------- #

run_time <- function(x){
  time <- bench_time(x)
  mem <- bench_memory(x)
  obj <- list("run_time" = time, "memory" = mem)
  return(obj)
}
