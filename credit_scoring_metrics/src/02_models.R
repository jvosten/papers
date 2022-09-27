
# Models *********************************************************************************************************************** #
# 
# The entire modeling process is organized within the tidymodels framework. Every classification model follows three steps:
# - the MODELTYPE_mod variable contains the model specification
# - the MODELTYPE_grid variable contains the corresponding tuning grid
# - the MODELTYPE_workflow variable bundles model, grid and preprocessing into a workflow
# For more detailed information see https://www.tidymodels.org/start/recipes/
#
# For an overview of a model call MODELTYPE_workflow, e.g. rf_workflow
#
# Logistic Regression Model **************************************************************************************************** #

lr_mod <- 
  parsnip::logistic_reg(penalty = tune(), mixture = 1) %>% # parsnips ›penalty‹ equals glmnets ›lamda‹; ›mixture‹ equals ›alpha‹
  parsnip::set_engine("glmnet") %>%
  translate()

lr_grid <- tibble(penalty = 10^seq(-4, -1, length.out = 25))

lr_workflow <- 
  workflow() %>% 
  add_model(lr_mod) %>% 
  add_recipe(preprocess)

# Random Forest Model ********************************************************************************************************** #

rf_mod <- 
  rand_forest(mtry = tune(), min_n = tune(), trees = 1000) %>% 
  set_engine("ranger") %>% 
  set_mode("classification") %>% 
  translate()

rf_grid <- rf_mod %>%
  parameters() %>%
  finalize(select(train_data, -default)) %>%
  grid_max_entropy(size = 25) 

rf_workflow <- 
  workflow() %>% 
  add_model(rf_mod) %>% 
  add_recipe(preprocess)

# Support Vector Machine ******************************************************************************************************* #

svm_mod <-
  svm_rbf(cost = tune(), rbf_sigma = tune()) %>%
  set_mode("classification") %>%
  set_engine("kernlab") %>%
  translate()

svm_grid <- svm_mod %>%
  parameters() %>%
  update(cost = cost(c(-2, 7)), 
         rbf_sigma = rbf_sigma(c(-3, -1))) %>% 
  grid_random(size = 25) 

svm_workflow <- 
  workflow() %>% 
  add_model(svm_mod) %>% 
  add_recipe(preprocess)

# Multilayer Perceptron ******************************************************************************************************** #

mlp_mod <-
  mlp(hidden_units = tune(), penalty = tune(), activation = "relu") %>%
  set_mode("classification") %>%
  set_engine("keras") %>%
  translate()

mlp_grid <- mlp_mod %>%
  parameters() %>%
  update(penalty = penalty(c(-4, -1))) %>%
  grid_max_entropy(size = 25)

mlp_workflow <-
  workflow() %>%
  add_model(mlp_mod) %>%
  add_recipe(preprocess)

# Boosted Tree Model *********************************************************************************************************** #

xgb_mod <-
  boost_tree(mtry = tune(), tree = tune(),
             learn_rate = tune(), tree_depth = tune()) %>%
  set_engine("xgboost") %>%
  set_mode("classification")%>%
  translate()

xgb_grid <- xgb_mod %>%
  parameters() %>%
  update(mtry = mtry(c(2L, 16L)),
         trees = trees(c(1000L, 2000L)),
         tree_depth = tree_depth(c(1L, 5L)),
         learn_rate = learn_rate(c(-2, -0.01))) %>% 
  finalize(select(train_data, -default)) %>%
  grid_random(size = 25) 

xgb_workflow <- 
  workflow() %>% 
  add_model(xgb_mod) %>% 
  add_recipe(preprocess)

# Naive Bayes ****************************************************************************************************************** #

nb_mod <- 
  naive_Bayes(smoothness = tune(), Laplace = tune()) %>% 
  set_mode("classification") %>%
  set_engine("klaR") %>% 
  translate()

nb_grid <- nb_mod %>%
  parameters() %>%
  grid_max_entropy(size = 25) 

#arrange(nb_grid, smoothness) %>% print(n =25) 

nb_workflow <- 
  workflow() %>% 
  add_model(nb_mod) %>% 
  add_recipe(preprocess)

# Fitting the models *********************************************************************************************************** #
#
# The fit is performed for each data set configuration, regarding the setting of preprocess, resulting in three 
# trained_models_DATASETTYPE objectcs, which are then stored in a list. In case the result file is needed, see README for 
# instructions on how to obtain. [In case you load the trained_models objects from "credit_results.RData", make sure to bundle them
# as a single list object (s. line 271)]

workflow_list <- list(lr_workflow, rf_workflow, 
                      svm_workflow, mlp_workflow, 
                      xgb_workflow, nb_workflow)
grid_list <- list(lr_grid, rf_grid, 
                  svm_grid, mlp_grid, 
                  xgb_grid, nb_grid)


trained_models_DATASETTYPE <- map2(.x = workflow_list,
                                   .y = grid_list,
                                   ~tune_grid(.x , 
                                              resamples = val_data, 
                                              grid = .y, 
                                              control = control_grid(save_pred = TRUE, verbose = FALSE),
                                              metrics = our_metrics)) %>%
  set_names(nm = c("LR", "RF", "SVM", "MLP", "XGB", "NB"))


# save(trained_models_raw,
#      trained_models_smote,
#      trained_models_down,
#      file = here("out", "credit_results.RData"))

trained_models <- list("Raw" = trained_models_raw, 
                       "Down" = trained_models_down, 
                       "SMOTE" = trained_models_smote) %>%
  map(enframe)
