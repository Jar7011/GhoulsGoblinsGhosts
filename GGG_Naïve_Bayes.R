# Load necessary libraries
library(tidyverse)
library(tidymodels)
library(vroom)
library(doParallel)
library(discrim)
library(themis)

## Naïve Bayes ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change output class to factor
train_data <- train_data %>% 
  mutate(type = as.factor(type))

# Create recipe
nb_recipe <- recipe(type ~ ., data = train_data) %>% 
  step_mutate_at(all_nominal_predictors(), fn = factor) %>% 
  step_lencode_glm(all_nominal_predictors(), outcome = vars(type)) %>% 
  step_range(all_numeric_predictors(), min=0, max=1) %>% 
  step_smote(all_outcomes(), neighbors = 3)

# Create model 
nb_model <- naive_Bayes(Laplace = tune(), smoothness = tune()) %>% 
  set_engine('naivebayes') %>% 
  set_mode('classification')

# Set workflow
nb_wf <- workflow() %>% 
  add_recipe(nb_recipe) %>% 
  add_model(nb_model)

# Grid of values to tune over
nb_grid_params <- grid_regular(Laplace(), smoothness(), levels = 10)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Run the CV
cv_results <- nb_wf %>% 
  tune_grid(resamples = folds,
            grid = nb_grid_params,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow
final_wf <- nb_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
nb_preds <- predict(final_wf, new_data = test_data, type = "class") %>% 
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = nb_preds, file = "./Naive_Bayes.csv", delim = ",")
