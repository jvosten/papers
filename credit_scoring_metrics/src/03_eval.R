
# Obtain results *************************************************************************************************************** #
#
# Obtain the best metrics -------------------------------------------------------------------------------------------------------#
#
best_metrics_list <- function(tml){ # tml = trained models list
  
  max_metric <- function(model_list){
    model_list %>%
      mutate(
        max_val = map2(.metric, data, ~ summarise(.y, max(mean, na.rm = TRUE)))
      ) %>%
      unnest(cols = max_val) %>% 
      select(-data)
  }
  
  bml <- function(tml){
    metric_res <- tml %>%
      deframe() %>%
      map(collect_metrics) %>%
      map(group_by, .metric) %>%
      map(nest) 
    
    best_metrics <- metric_res %>%
      map(., max_metric) %>%
      map2(.y = names(metric_res), ~rename(.x, !!.y := `max(mean, na.rm = TRUE)`)) %>%
      flatten_dfr(.id = NULL) %>% 
      rename(Metric = .metric)
  }
  beml <- map(tml, bml)  
  
  return(beml)
}

bm_list <- best_metrics_list(trained_models)

# Table for mean difference per classifier for each pair of data sets -----------------------------------------------------------# 

metrics_diff <- function(bm1, bm2){
  (bm1 %>% column_to_rownames(var = "Metric")  - bm2 %>% column_to_rownames(var = "Metric")) %>% 
    rownames_to_column(var = "Avg distance for:") %>%
    rowwise() %>%
    mutate(Mean = mean(c(LR, RF, SVM, MLP, XGB, NB))) %>%
    select("Avg distance for:", Mean)
}

mean_diff <-
  map2(.x = list(bm_list$Raw, bm_list$Raw, bm_list$SMOTE), 
       .y = list(bm_list$Down, bm_list$SMOTE, bm_list$Down), 
       ~metrics_diff(bm1 = .x, bm2 = .y)) %>%
  map2(.y = list("Raw & Down", 
                 "Raw & SMOTE", 
                 "SMOTE & Down"), ~rename(.x, !!.y := Mean)) %>%
  flatten_dfr()

# Obtain the curves -------------------------------------------------------------------------------------------------------------#

# ROC curves for all three data sets stored in a list object, each individual curve obtained through pr_list$TYPE
roc_curves_list <- function(tml){
  
  rcl <- function(tml){
    roc_best <- tml %>%
      deframe() %>%
      map(select_best, metric = "roc_auc", n = 1) 
    
    roc_curves <- tml %>%
      deframe() %>%
      map2(., .y = roc_best, ~ collect_predictions(.x, parameters = .y)) %>% 
      map(roc_curve, truth = default, estimate = .pred_1) %>%
      map2(.y = names(deframe(tml)), ~ mutate(.x, model = .y))
    
    roc_res <- map_df(roc_curves, bind_rows) %>% 
      ggplot(aes(x = 1 - specificity, y = sensitivity, col = model)) + 
      geom_path(lwd = 1.5, alpha = 0.8) +
      geom_abline(lty = 3) + 
      coord_equal() + 
      scale_color_viridis_d(option = "plasma", end = 0.8)
  }
  roccl <- map(tml, rcl) 
  
  return(roccl)
}

roc_list <- roc_curves_list(trained_models)

# Precision Recall curves for all three data sets stored in a list object, obtained through pr_list$TYPE

pr_curves_list <- function(tml){
  
  pcl <- function(tml){
    pr_best <- tml %>%
      deframe() %>%
      map(select_best, metric = "pr_auc", n = 1) 
    
    pr_curves <- tml %>%
      deframe() %>%
      map2(., .y = pr_best, ~ collect_predictions(.x, parameters = .y))%>% 
      map(pr_curve, truth = default, estimate = .pred_1) %>%
      map2(.y = names(deframe(tml)), ~ mutate(.x, model = .y))
    
    prc_res <- map_df(pr_curves, bind_rows) %>%
      ggplot(aes(x = recall, y = precision, col = model)) + 
      geom_path(lwd = 1.5, alpha = 0.8) +
      #geom_abline(lty = 3) + 
      coord_equal() + 
      ggtitle((paste("ROC curves for", names(tml), "data"))) + 
      scale_color_viridis_d(option = "plasma", end = 0.8) +
      expand_limits(y = c(0,1))
  }
  prcl <- map(tml, pcl)
  
  return(prcl)
}

pr_list <- pr_curves_list(trained_models)

# Evaluating the models ******************************************************************************************************** #
#
# Prepare data for statitical tests ---------------------------------------------------------------------------------------------#
to_test <- function(mt){
  
  test <- function(bm, metric) {
    if(metric == "accuracy"){
      slice(bm, 1)
    } else {
      slice(bm, 8)
    }
  }
  
  bm_list %>%
    map(., ~test(bm = .x, metric = mt)) %>%
    bind_rows(.id = "Dataset") %>% 
    select(-2) %>%
    column_to_rownames(var = "Dataset")
}

test_data <-
  map(c("roc_auc", "accuracy"), to_test) %>% 
  set_names(nm = c("AUC", "Accuracy"))

# Friedman alligned rank test ---------------------------------------------------------------------------------------------------# 
test_data %>% 
  map(friedmanAlignedRanksTest) %>%
  map(broom::tidy) %>%
  map2(.y = names(test_data), ~ mutate(.x, Metric = .y)) %>%
  bind_rows() %>%
  select(Metric, 1:3)

# Friedman post hoc test --------------------------------------------------------------------------------------------------------# 
post_hoc <- function(metric = "AUC"){
  test_data[[metric]] %>%
    friedmanAlignedRanksPost(.) %>%
    plotPvalues(., alg.order=c(6:1)) + 
    ggtitle((paste("Friedman p-values for", metric))) + 
    scale_fill_viridis(direction = 1, option = "C")
}

grid.arrange(eval(post_hoc("AUC")), eval(post_hoc("Accuracy")), layout_matrix = rbind(c(1,2)))

# Save results ***************************************************************************************************************** #
#
# save(bm_list,
#      roc_list, 
#      pr_list,
#      test_data,
#      mean_diff,
#      file = here("out","exp_results.RData")
#
# load(here("out", "exp_results.RData") # In case this file is needed, see README for instructions on how to obtain
# 