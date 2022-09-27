
# ****************************************************** IMPORTANT ************************************************************* #


# Code requires a less recent version of dplyr then 1.0.0 and rsample version 0.0.6, due to the major dplyr update in June 
# (dependencies in other packages have not been actualized yet); otherwise the code will not function. 


# Getting started ************************************************************************************************************** #
pkgs = c("skimr", "viridis", "corrr", "gridExtra", "grid",
         "scmamp", "glmnet", "klaR","rpart", 
         "kernlab", "keras", "xgboost", "discrim", 
         "tidyverse", "tidymodels") 
lapply(pkgs, library, character.only = TRUE, quietly = TRUE) 
set.seed(4321)
theme_set(theme_light())

setwd("~/src/git/papers/credit_scoring_metrics")
library(here)

data_raw <- 
  read_csv(here("data", "semWI_credit.csv"))
# load("credit_results.RData") # In case this file is needed, see README for instructions on how to obtain

# Overview and data preparation ************************************************************************************************ #

skim(data) %>%
  skimr::focus(n_missing, numeric.mean, numeric.sd) %>%
  select(-1) 

# Barplot default Var
bplot <- function(dataset){
  dataset %>% 
    ggplot(aes(x = DEFAULT, y = (..count..)/sum(..count..)), colour = DEFAULT) +
    geom_bar(fill = "#4271AE",
             alpha = 0.8) +
    scale_y_continuous(labels = scales::percent) +
    labs(title = "", y = "Percent", x = "default")
}

# Correlation plot
corplot <- function(dataset){
  dataset %>%
    correlate %>%
    focus(DEFAULT) %>%
    mutate(rowname = reorder(rowname, DEFAULT)) %>%
    ggplot(aes(rowname, DEFAULT)) +
    geom_col(fill = "#4271AE") + coord_flip() +
    expand_limits(y = c(-1,1))
}

grid.arrange(eval(bplot(data_raw)), eval(corplot(data_raw)), layout_matrix = rbind(c(1,2)))

# Data wrangling
data <- data_raw %>%
  select(-PROFIT) %>%
  mutate(
    DEFAULT = factor(DEFAULT),
    DEFAULT = fct_relevel(DEFAULT, "1"), # fct level is recoded for yardstick package metrics
  ) %>%
  rename(unsecure_lines = RevolvingUtilizationOfUnsecuredLines) %>%
  rename(nr_past_due30 = NumberOfTime30.59DaysPastDueNotWorse) %>% 
  rename(nr_open_credits = NumberOfOpenCreditLinesAndLoans) %>%
  rename(nr_re_loans = NumberRealEstateLoansOrLines) %>%
  rename(nr_family = NumberOfDependents) %>%
  rename_all(tolower)

# Splitting data
#
# Stratified split
data_split <- initial_split(data, prop = 3/4, strata = default)

train_data <- training(data_split)
test_data <- testing(data_split)

val_data <- validation_split(train_data, 
                             strata = default, 
                             prop = 0.80)
# Check stratification
train_data %>% 
  count(default) %>% 
  mutate(prop = n/sum(n))

test_data %>% 
  count(default) %>% 
  mutate(prop = n/sum(n))

# Preprocessing for training data
#
# The recipe package offers the option to bundle all of processing for a data set (normalization, dummies, etc.); since our data
# does not require much processing, we use it to create the three different data sets via the themis package.
# We want to obtain three different data sets: one raw data set, without correction of the class imbalance; on data set with 
# correction through down-sampling and one with SMOTE sampling. To create Each of this data set we executed the whole modeling 
# process three times, for trained_models_raw we excluded the themis::step_ function; for downsampling we altered the preprocess 
# variable by activating themis::step_downsample(all_outcomes()); the same for SMOTE with themis::step_downsample(all_outcomes()).
# 
# The default set up for preprocess includes the downsample step, since this configuration requires the least computation time
#
preprocess <- 
  recipes::recipe(default ~ ., data = train_data) %>% # Determine outcome (default) and predictor variables ("." = all other vars):  
  themis::step_downsample(all_outcomes())

# Controll the preprocessing
juice(prep(preprocess)) %>%
count(default) %>% 
  mutate(prop = n/sum(n))

# Set metrics
our_metrics <- 
  yardstick::metric_set(
    roc_auc,
    pr_auc,
    average_precision,
    precision,
    recall,
    kap,
    f_meas,
    accuracy
    )
