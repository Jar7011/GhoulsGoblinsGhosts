library(tidymodels)
library(vroom)
library(themis)
library(embed)

## MLP ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')

# Change output class to factor
train_data <- train_data %>% 
  mutate(type = as.factor(type))

# Create recipe
mlp_recipe <- recipe(type ~ ., data = train_data) %>% 
  step_mutate(color = factor(color)) %>% 
  step_dummy(color) %>% 
  step_range(all_numeric_predictors(), min=0, max=1)

# Create model 
mlp_model <- mlp(hidden_units = tune(),
                 epochs = 250) %>% 
  set_engine('keras') %>% 
  set_mode('classification')

# Create workflow
mlp_wf <- workflow() %>% 
  add_recipe(mlp_recipe) %>% 
  add_model(mlp_model)

# Grid of values to tune over
tuning_grid <- grid_regular(hidden_units(range = c(1,20)),
                            levels = 5)

# Split data for CV
folds <- vfold_cv(train_data, v = 10, repeats = 1)

# Run the CV
cv_results <- mlp_wf %>% 
  tune_grid(resamples = folds,
            grid = tuning_grid,
            metrics = metric_set(accuracy))

# Cross validation plot
my_plot <- cv_results %>% 
  collect_metrics() %>% 
  filter(.metric=="accuracy") %>% 
  ggplot(aes(x=hidden_units, y=mean)) + geom_line()

# Save plot
ggsave(filename = "plot.jpeg", plot = my_plot, device = "jpeg")

# Find best tuning params
best_tuning_params <- cv_results %>% 
  select_best(metric = 'accuracy')

# Finalize workflow
final_wf <- mlp_wf %>% 
  finalize_workflow(best_tuning_params) %>% 
  fit(data = train_data)

# Predict and format for submission
mlp_preds <- predict(final_wf, 
                     new_data = test_data,
                     type = "class") %>% 
  bind_cols(., test_data) %>% 
  select(id, .pred_class) %>% 
  rename(type = .pred_class)

# Write out the file
vroom_write(x = knn_preds, file = "./MLP.csv", delim = ",")