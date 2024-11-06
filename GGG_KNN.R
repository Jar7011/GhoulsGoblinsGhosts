library(tidymodels)
library(doParallel)
library(vroom)
library(themis)
library(embed)

## KNN ##
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change output class to factor
train_data <- train_data %>% 
  mutate(type = as.factor(type))

# Create recipe
knn_recipe <- recipe(type ~ ., data = train_data) %>% 
  step_mutate(color = factor(color)) %>% 
  step_lencode_glm(color, outcome = vars(type)) %>% 
  step_normalize(all_numeric_predictors())

# Create model
knn_model <- nearest_neighbor(neighbors = tune()) %>% 
  set_mode('classification') %>% 
  set_engine('kknn')

# Create workflow
knn_wf <- workflow() %>% 
  add_recipe(knn_recipe) %>% 
  add_model(knn_model)

# Grid of values to tune over
tuning_grid <- grid_regular(neighbors(),
                            levels = 10)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Set up parallel computing
num_cores <- detectCores()
cl <- makePSOCKcluster(num_cores)
registerDoParallel(num_cores)

# Run the CV
cv_results <- knn_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(roc_auc))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'roc_auc')

# Finalize workflow
final_wf <- knn_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
knn_preds <- predict(final_wf, 
                     new_data = test_data,
                     type = "class") %>% 
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class)

# End parallel computing
stopCluster(cl)

# Write out the file
vroom_write(x = knn_preds, file = "./KNN.csv", delim = ",")

# Score with accuracy metric: 0.70510
# Score with roc_auc metric: 0.70699

