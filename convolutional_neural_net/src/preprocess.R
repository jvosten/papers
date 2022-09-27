library(tidyverse)
library(keras)
library(tfdatasets)
library(tfhub)
library(bench)

setwd("~/path/to/papers/convolutional_neural_net")
library(here)


# Calls helper functions
helpers <- function() {
  old <- setwd(here())
  on.exit(setwd(old), add = TRUE)
  
  setwd(here("src"))
  source("helper_funs.R")
}
helpers()
set.seed(4321)
theme_set(theme_light())
reticulate::py_discover_config()
# tf_config()
load(here("out","results.RData"))

# Preprocesing ***************************************************************************************************************** #

# Obtain data and create directories ------------------------------------------------------------------------------------------- # 

# Original data set was downloaded from kaggle.com:
# https://www.kaggle.com/gpiosenka/100-bird-species

# As the author updates the data set frequently, the version of it (containing 210 classes), can be obtained from:
# HU Box

# All paths in this script are relative to the parent directory "/paper_js"; viz. it is recommended to move the downloaded image 
# data to the path "/paper/data"

# Defining paths
train_path <- "./data/train"
test_path <- "./data/test"
valid_path <- "./data/valid"

# Create directories for small sample (25 classes) and tiny sample (5 classes)
base_dir <- "./data/small_sample"
dir.create(base_dir)
sub_dirs(base_dir)

base_dir <- "./data/tiny_sample"
dir.create(base_dir)
sub_dirs(base_dir)

# Create directory for saving output
output_dir <- "./output"
dir.create(output_dir)

#unlink(base_dir, recursive = TRUE)

# As soon as these directories are created, the above commands can be commented; for furter use it is sufficient to just call 
# sample_path()

# ****************************************************** IMPORTANT ************************************************************* #

# If you run tensorflow on a GPU, you can just use all of the classes of the data set; skip the sampling process and switch the 
# sample_path function to "orig. If tensorflow is only set up with CPU, use the sampling function to obtain a smaller data sets 
# with less classes, otherwise you might experience excessive computation time for the model.

# Fill the smaller sample directories with randomly sampled classes

# Small data set
sample_path("small")
sampling(25)

# Tiny data set
sample_path("tiny")
sampling(5)

# Overview and data set preparation ******************************************************************************************** #

# See the effect of image augmentation
aug_example_pic()

# Save picture
# png(file="aug_ex.png")
# aug_example_pic()
# dev.off()

# Data set preparation --------------------------------------------------------------------------------------------------------- # 

# As most models operate on the following parameters, we set them as a default in the wrapper function for the tensor producing
# data set generator function
sample_path("tiny")
image_shape <- c(150, 150, 3) # Consists of img_width, img_height and (color) channels
target_size <- image_shape[-3]
batch_size <- 32 
classes_update <- function() list.dirs(train_dir, full.names = FALSE, recursive = FALSE) 

# This function transforms an image into a tensor and rescales it
datagen <- image_data_generator(rescale = 1/255) 

# This is a wrapper function for the flow_images_from_directory function, which infinitely loops over a directory, reads all the 
# images from that directory in batches and transforms them to a tensor
data_generator <- function(...) {
  call <- as.list(match.call())[-1] # first position is the function_name
  
  custom_args <- list(generator = datagen,
                      target_size = target_size,
                      batch_size = batch_size) 
  
  overlap_args <-  names(call) %in% names(custom_args) # handle overlapping args
  if (!any(overlap_args)) call <- c(call, custom_args)
  
  do.call(flow_images_from_directory, call) # exectue flow_images_from_directory() with the custom settings
}

# Test, if data_generator function works: create a data set with augmented pictures and print it
test_data <- data_generator(train_path)
test_pic(test_data)


# See the class distribution
class_hist <- test_data$classes %>% 
  factor(.) %>%
  as_tibble(.) %>%
  count(value) %>%
  ggplot(., aes(n)) +
    geom_histogram(bins = 30,
                   fill = "#4271AE", 
                   colour = "#4271AE", 
                   alpha = 0.8) +
    labs(x = "# of Images", y = "# of Classes") +
  geom_vline(aes(xintercept=mean(n, na.rm=T)),   
             color="red", linetype="dashed", size=1)

rm(test_data)

# **************************************************** Data Set Generation ***************************************************** #
#
# All further data set generation steps are placed before each model in models.R, to maintain a better overview.
#
# ****************************************************************************************************************************** #
