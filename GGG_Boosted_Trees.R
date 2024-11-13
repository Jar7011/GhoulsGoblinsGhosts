library(tidymodels)
library(vroom)
library(themis)
library(embed)
library(bonsai)

## Boosted Trees ## 

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change output class to factor
train_data <- train_data %>% 
  mutate(type = as.factor(type))

# Create recipe
boost_recipe <- recipe(type ~ ., data = train_data) %>% 
  step_mutate(color = factor(color)) %>% 
  step_dummy(color) %>% 
  step_normalize(all_numeric_predictors())

# Create model 
boost_model <- boost_tree(tree_depth = tune(),
                          trees = tune(),
                          learn_rate = tune()) %>% 
  set_engine('lightgbm') %>% 
  set_mode('classification')

# Create workflow
boost_wf <- workflow() %>% 
  add_recipe(boost_recipe) %>% 
  add_model(boost_model)

# Grid of values to tune over
tuning_grid <- grid_regular(tree_depth(), trees(), learn_rate(), levels = 5)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Run the CV
cv_results <- boost_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'accuracy')

# Finalize workflow
final_wf <- boost_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
boost_preds <- predict(final_wf, new_data = test_data, type = "class") %>% 
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class)

# Write out the file
vroom_write(x = boost_preds, file = "./Boost_Trees.csv", delim = ",")
