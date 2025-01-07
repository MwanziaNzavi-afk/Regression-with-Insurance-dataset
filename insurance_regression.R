# Load Required Libraries
library(tidyverse)    # For data manipulation and visualization
library(caret)        # For data splitting and machine learning
library(randomForest) # For Random Forest model
library(xgboost)      # For XGBoost model
library(Metrics)      # For RMSLE calculation

# Set File Paths
train_path <- "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/train.csv"
test_path <- "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/test.csv"
submission_path <- "C:/Users/mwanz/OneDrive/Desktop/Regression with an Insurance Dataset/sample_submission.csv"

# Load Datasets
train_data <- read.csv(train_path)
test_data <- read.csv(test_path)
sample_submission <- read.csv(submission_path)

# Data Exploration
print("Structure of Train Data:")
str(train_data)

print("Structure of Test Data:")
str(test_data)

# Summary of the target variable
summary(train_data$Premium.Amount)

# Check for missing values
cat("Missing values in Train Data:\n")
print(colSums(is.na(train_data)))

cat("Missing values in Test Data:\n")
print(colSums(is.na(test_data)))

# Visualize the Target Variable
ggplot(train_data, aes(x = Premium.Amount)) +
  geom_histogram(bins = 30, fill = "blue", color = "white") +
  theme_minimal() +
  labs(title = "Distribution of Premium Amount", x = "Premium Amount", y = "Frequency")

# Log-transform the target variable
train_data$LogPremium <- log1p(train_data$Premium.Amount)

# Remove irrelevant columns (e.g., id)
train_data <- train_data %>% select(-id)
test_data <- test_data %>% select(-id)

# Handle missing values (replace with mean for simplicity)
train_data[is.na(train_data)] <- lapply(train_data, function(x) ifelse(is.numeric(x), mean(x, na.rm = TRUE), x))
test_data[is.na(test_data)] <- lapply(test_data, function(x) ifelse(is.numeric(x), mean(x, na.rm = TRUE), x))

# Normalize numeric features
numeric_features <- sapply(train_data, is.numeric)
train_data[numeric_features] <- scale(train_data[numeric_features])
test_data[numeric_features] <- scale(test_data[numeric_features])

# Split train data into training and validation sets
set.seed(123)
train_index <- createDataPartition(train_data$LogPremium, p = 0.8, list = FALSE)
train_set <- train_data[train_index, ]
valid_set <- train_data[-train_index, ]

# Define RMSLE function
rmsle <- function(pred, actual) {
  sqrt(mean((log1p(pred) - log1p(actual))^2))
}

# Train Random Forest Model
rf_model <- randomForest(LogPremium ~ ., data = train_set, ntree = 100, mtry = 5)
rf_predictions <- expm1(predict(rf_model, valid_set))
rf_rmsle <- rmsle(rf_predictions, expm1(valid_set$LogPremium))

# Train XGBoost Model
xgb_model <- xgboost(data = as.matrix(train_set %>% select(-LogPremium)),
                     label = train_set$LogPremium,
                     nrounds = 100,
                     objective = "reg:squarederror",
                     verbose = 0)
xgb_predictions <- expm1(predict(xgb_model, as.matrix(valid_set %>% select(-LogPremium))))
xgb_rmsle <- rmsle(xgb_predictions, expm1(valid_set$LogPremium))

# Compare RMSLE
cat("Random Forest RMSLE:", rf_rmsle, "\n")
cat("XGBoost RMSLE:", xgb_rmsle, "\n")

# Generate predictions on the test set using the best model (e.g., Random Forest)
test_predictions <- expm1(predict(rf_model, test_data))

# Create submission file
submission <- sample_submission
submission$Premium.Amount <- test_predictions
write.csv(submission, "submission.csv", row.names = FALSE)

cat("Submission file created successfully as 'submission.csv'.\n")
