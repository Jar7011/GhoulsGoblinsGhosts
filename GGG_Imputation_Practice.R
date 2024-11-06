library(tidyverse)
library(tidymodels)
library(vroom)
library(DataExplorer)

## Imputation Practice ##

# Read in data
train_data <- vroom('train.csv')
test_data <- vroom('test.csv')
missing_train_data <- vroom('trainWithMissingValues.csv')

# Create recipe
imputation_recipe <- recipe(type ~ ., data = missing_train_data) %>% 
  step_mutate(color = factor(color)) %>% 
  step_impute_knn(all_of(c('hair_length', 'rotting_flesh', 'bone_length')), 
                  impute_with = imp_vars(all_predictors()),
                  neighbors = 6)

# Prep and bake recipe
baked_recipe <- imputation_recipe %>% 
  prep() %>% 
  bake(new_data = missing_train_data)

# Get rmse of the imputations
rmse <- rmse_vec(train_data[is.na(missing_train_data)],
         baked_recipe[is.na(missing_train_data)])

# 0.1456145