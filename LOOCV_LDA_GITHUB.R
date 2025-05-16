#### LOOCV with LDA ###

#Create Dummy Dataset to work with. 
set.seed(123)  # For reproducibility

# Create outcome variable
Y <- c(rep(1, 500), rep(0, 500))
num_obs <- 1000
num_vars <- 300
num_informative <- 50

# Randomly choose informative predictor indices
informative_vars <- sample(1:num_vars, num_informative)
non_informative_vars <- setdiff(1:num_vars, informative_vars)

# Initialize matrix for predictors
X <- matrix(nrow = num_obs, ncol = num_vars)

# Buffer to ensure positive shift
buffer <- 0.01

# Generate non-informative predictors (unrelated to Y)
for (i in non_informative_vars) {
  mean_i <- runif(1, -10, 10)
  sd_i <- runif(1, 1, 5)
  values <- rnorm(num_obs, mean = mean_i, sd = sd_i)
  
  # Add outliers
  outlier_idx <- sample(1:num_obs, size = round(0.1 * num_obs))
  values[outlier_idx] <- values[outlier_idx] + rnorm(length(outlier_idx), mean = 0, sd = 10)
  
  # Shift to make all values positive
  values <- values - min(values) + buffer
  
  X[, i] <- values
}

# Generate informative predictors (distribution varies by Y)
for (i in informative_vars) {
  mean_pos <- runif(1, -5, 5)
  sd_pos <- runif(1, 1, 3)
  mean_neg <- mean_pos + runif(1, -3, 3)
  sd_neg <- sd_pos + runif(1, -1, 1)
  
  values <- numeric(num_obs)
  values[Y == 1] <- rnorm(sum(Y == 1), mean = mean_pos, sd = sd_pos)
  values[Y == 0] <- rnorm(sum(Y == 0), mean = mean_neg, sd = sd_neg)
  
  # Add outliers
  outlier_idx <- sample(1:num_obs, size = round(0.1 * num_obs))
  values[outlier_idx] <- values[outlier_idx] + rnorm(length(outlier_idx), mean = 0, sd = 10)
  
  # Shift to make all values positive
  values <- values - min(values) + buffer
  
  X[, i] <- values
}

# Create data frame
X_df <- as.data.frame(X)
colnames(X_df) <- paste0("P", 1:num_vars)

# Combine with outcome
df <- cbind(Y = Y, X_df)

# Create Observation IDs
obs_id <- c(paste0("POS", 1:500), paste0("NEG", 1:500))

# Add to data frame
df$ID <- obs_id

# Optionally move ID column to the front
df <- df[, c("ID", "Y", paste0("P", 1:num_vars))]

write.csv(df, "Dummy_data.csv")

#transpose df to explore means for each predictor. 
df_t <- t(df)

df_t <- as.data.frame(df_t)

# 1. Set first row as column names
colnames(df_t) <- as.character(df_t["ID", ])

# 2. Remove the ID row
df_t <- df_t[-which(rownames(df_t) == "ID"), ]

# Convert all values to numeric (column-wise)
df_t <- as.data.frame(apply(df_t, 2, as.numeric))

# Optional: set row names back if lost
rownames(df_t) <- c("Y", paste0("P", 1:300))

# Check the result
head(df_t[, 1:4])

df_t <- df_t[rownames(df_t) != "Y", ]

# Get means and SDs for POS (cols 2–51)
pos_means <- apply(df_t[, 1:500], 1, mean)
pos_sds   <- apply(df_t[, 1:500], 1, sd)

# Get means and SDs for NEG (cols 52–101)
neg_means <- apply(df_t[, 501:1000], 1, mean)
neg_sds   <- apply(df_t[, 501:1000], 1, sd)

# Combine into one summary data frame
summary_df <- data.frame(
  Predictor = rownames(df_t),
  POS_Mean = pos_means,
  POS_SD = pos_sds,
  NEG_Mean = neg_means,
  NEG_SD = neg_sds
)

# View the first few rows
head(summary_df)

# Preallocate vectors for p-values
t_test_p <- numeric(nrow(df_t))
kruskal_p <- numeric(nrow(df_t))

# Loop over each row (predictor)
for (i in 1:nrow(df_t)) {
  row_values <- as.numeric(df_t[i, ])  # current row
  group <- rep(c("POS", "NEG"), each = 500) # group labels
  
  # t-test
  t_test_result <- t.test(row_values[1:500], row_values[501:1000])
  t_test_p[i] <- t_test_result$p.value
  
  # Kruskal-Wallis test
  kruskal_result <- kruskal.test(row_values ~ group)
  kruskal_p[i] <- kruskal_result$p.value
}

# Add to summary table
summary_df$t_test_p <- t_test_p
summary_df$kruskal_p <- kruskal_p

# View top rows sorted by significance
summary_df_sorted <- summary_df[order(summary_df$t_test_p), ]
head(summary_df_sorted, 10)

#####Ploting the most significant#####
library(tidyr)
library(dplyr)

top20 <- summary_df %>%
  arrange(t_test_p) %>%
  slice(1:20)

top20 <- as.data.frame(top20)

top20_long <- top20 %>%
  dplyr::select(Predictor, POS_Mean, NEG_Mean, POS_SD, NEG_SD) %>%
  tidyr::pivot_longer(
    cols = c(POS_Mean, NEG_Mean, POS_SD, NEG_SD),
    names_to = c("Group", ".value"),
    names_pattern = "(POS|NEG)_(.*)"
  )

library(ggplot2)

ggplot(top20_long, aes(x = reorder(Predictor, -Mean), y = Mean, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD),
                position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Top 20 Predictors by Significance",
       x = "Predictor", y = "Mean ± SD") +
  theme_minimal() +
  theme(axis.text.x = element_text(angle = 45, hjust = 1))

#####Train and hold out####
#here we now have the significant predictors but none display a perfect seporation.
#it is possible to utilise LASSO regression to shrink the posible number of predictors

# Load required libraries
library(glmnet)

df_test <- df[, 2:302]

# Predictor matrix
X <- as.matrix(df_test[, 2:301])

# Outcome vector
Y <- df_test[, 1]

# Fit LASSO logistic regression without cross-validation
lasso_fit <- glmnet(X, Y, family = "binomial", alpha = 1)

plot(lasso_fit, xvar = "lambda", label = TRUE)

# View all lambda values used
lasso_fit$lambda

#plot the number of predictors by lamba
num_selected <- sapply(lasso_fit$lambda, function(lam) {
  sum(coef(lasso_fit, s = lam)[-1, ] != 0)  # exclude intercept
})

plot(lasso_fit$lambda, num_selected, type = "b", log = "x",
     xlab = "Lambda (log scale)", ylab = "Number of selected predictors")


# choose the lamba you want, e.g. lasso_fit$lambda[22]
coefs <- coef(lasso_fit, s = lasso_fit$lambda[4])

# Get selected variable names
selected <- rownames(coefs)[which(coefs != 0)]
selected <- setdiff(selected, "(Intercept)")

length(selected)
print(selected)

library(tidyverse)

selected_vars <- c("P81",  "P94",  "P143", "P244", "P256", "P292")

# Reshape the data
summary_long <- summary_df %>%
  filter(Predictor %in% selected_vars) %>%
  pivot_longer(cols = c(POS_Mean, NEG_Mean, POS_SD, NEG_SD),
               names_to = c("Group", ".value"),
               names_pattern = "(POS|NEG)_(Mean|SD)")

ggplot(summary_long, aes(x = Predictor, y = Mean, fill = Group)) +
  geom_bar(stat = "identity", position = position_dodge(width = 0.9)) +
  geom_errorbar(aes(ymin = Mean - SD, ymax = Mean + SD),
                position = position_dodge(width = 0.9), width = 0.25) +
  labs(title = "Mean and SD of Selected Predictors by Group",
       y = "Mean Value ± SD", x = "Predictor")

###using LDA on the LASSO regression selection predictors 

library(MASS)

# Your selected predictors
selected_vars <- c("P81",  "P94",  "P143", "P244", "P256", "P292")

# Subset data with outcome + selected predictors
lda_data <- df_test[, c("Y", selected_vars)]

lda_fit <- lda(Y ~ ., data = lda_data)

print(lda_fit)
plot(lda_fit)

pred <- predict(lda_fit)

# Predicted classes
pred$class
pred$posterior

# Confusion table
table(Predicted = pred$class, Actual = lda_data$Y)

####Train and hold out#####
#now we have explored LASSO on the entire population and predicted back out using LDA
#it is time to complete a train and hold out. 
#training the data on a majority and testing it on an unseen minority.

library(glmnet)
library(MASS)
library(caret)
library(dplyr)

set.seed(123)  # for reproducibility

# Extract matrix and outcome
X <- as.matrix(df_test[, 2:301])
Y <- df_test$Y

# Results storage
results <- data.frame(Iteration = integer(),
                      Num_Selected = integer(),
                      Selected_Predictors = character(),
                      Sensitivity = numeric(),
                      Specificity = numeric(),
                      TP = integer(),
                      TN = integer(),
                      FP = integer(),
                      FN = integer(),
                      stringsAsFactors = FALSE)

for (i in 1:1000) {
  
  # Split data
  sample_indices <- sample(1:nrow(df_test), size = 0.8 * nrow(df_test))
  X_train <- X[sample_indices, ]
  Y_train <- Y[sample_indices]
  X_test <- X[-sample_indices, ]
  Y_test <- Y[-sample_indices]
  
  # Fit LASSO
  lasso_fit <- glmnet(X_train, Y_train, family = "binomial", alpha = 1)
  coefs <- coef(lasso_fit, s = lasso_fit$lambda[5])  # adjust lambda as needed
  selected <- rownames(coefs)[which(coefs != 0)]
  selected <- setdiff(selected, "(Intercept)")
  
  # Skip if no predictors selected
  if (length(selected) == 0) next
  
  # Prepare data frames
  train_df <- data.frame(Y = as.factor(Y_train), X_train[, selected, drop = FALSE])
  test_df  <- data.frame(Y = as.factor(Y_test),  X_test[, selected, drop = FALSE])
  
  # Fit LDA and predict
  lda_fit <- lda(Y ~ ., data = train_df)
  pred <- predict(lda_fit, newdata = test_df)$class
  
  # Confusion matrix
  cm <- confusionMatrix(pred, test_df$Y, positive = "1")
  cm_table <- cm$table
  TP <- cm_table["1", "1"]
  TN <- cm_table["0", "0"]
  FP <- cm_table["1", "0"]
  FN <- cm_table["0", "1"]
  
  # Store results
  results <- results %>%
    add_row(Iteration = i,
            Num_Selected = length(selected),
            Selected_Predictors = paste(selected, collapse = ", "),
            Sensitivity = cm$byClass["Sensitivity"],
            Specificity = cm$byClass["Specificity"],
            TP = TP, TN = TN, FP = FP, FN = FN)
}

# Summary across all iterations
summary_results <- results %>%
  summarise(Mean_Sensitivity = mean(Sensitivity, na.rm = TRUE),
            SD_Sensitivity = sd(Sensitivity, na.rm = TRUE),
            Mean_Specificity = mean(Specificity, na.rm = TRUE),
            SD_Specificity = sd(Specificity, na.rm = TRUE),
            Avg_Num_Selected = mean(Num_Selected),
            Total_TP = sum(TP), Total_TN = sum(TN),
            Total_FP = sum(FP), Total_FN = sum(FN))

print(results)
print(summary_results)

write.csv(results, "80_20_THO_results_1000it.csv")

###selected predictors plot

library(ggplot2)
library(dplyr)
library(tidyr)
library(stringr)

# 1. Split predictor strings into a list
predictor_list <- strsplit(results$Selected_Predictors, ",\\s*")

# 2. Unlist and create a data frame
all_selected <- unlist(predictor_list)
predictor_counts <- as.data.frame(table(all_selected))
colnames(predictor_counts) <- c("Predictor", "Count")

# 3. Sort by count
predictor_counts <- predictor_counts %>% arrange(desc(Count))

# 4. Plot
ggplot(predictor_counts, aes(x = reorder(Predictor, Count), y = Count)) +
  geom_bar(stat = "identity", fill = "steelblue") +
  coord_flip() +
  labs(title = "Frequency of Predictor Selection Across 1000 LASSO Iterations",
       x = "Predictor",
       y = "Count (Times Selected)")

####Histogram spreads of the predictions of the test samples. 

set.seed(123)

X <- as.matrix(df[, 3:302])
Y <- df$Y
IDs <- df$ID  # assuming you have this column

# Store posterior predictions
posterior_df <- data.frame(Iteration = integer(),
                           ID = character(),
                           True_Label = integer(),
                           Predicted_Prob = numeric(),
                           stringsAsFactors = FALSE)

for (i in 1:1000) {
  
  sample_indices <- sample(1:nrow(df), size = 0.8 * nrow(df))
  X_train <- X[sample_indices, ]
  Y_train <- Y[sample_indices]
  X_test <- X[-sample_indices, ]
  Y_test <- Y[-sample_indices]
  ID_test <- IDs[-sample_indices]
  
  lasso_fit <- glmnet(X_train, Y_train, family = "binomial", alpha = 1)
  coefs <- coef(lasso_fit, s = lasso_fit$lambda[22])  # or use cross-validation lambda
  selected <- rownames(coefs)[which(coefs != 0)]
  selected <- setdiff(selected, "(Intercept)")
  
  if (length(selected) == 0) next
  
  train_df <- data.frame(Y = as.factor(Y_train), X_train[, selected, drop = FALSE])
  test_df  <- data.frame(Y = as.factor(Y_test),  X_test[, selected, drop = FALSE])
  
  lda_fit <- lda(Y ~ ., data = train_df)
  posterior_probs <- predict(lda_fit, newdata = test_df)$posterior[, "1"]
  
  # Collect predictions
  iter_df <- data.frame(Iteration = i,
                        ID = ID_test,
                        True_Label = Y_test,
                        Predicted_Prob = posterior_probs)
  
  posterior_df <- bind_rows(posterior_df, iter_df)
}

# Review
head(posterior_df)

#Now to pick and plot 10 examples of the spread

library(dplyr)

# 1. Summarise posterior probabilities per sample ID
posterior_summary <- posterior_df %>%
  group_by(ID, True_Label) %>%
  summarise(Mean_Posterior = mean(Predicted_Prob, na.rm = TRUE),
            SD_Posterior = sd(Predicted_Prob, na.rm = TRUE),
            .groups = "drop")

# 2. Arrange in descending order (if looking at probability of being class 1)
posterior_summary <- posterior_summary %>%
  arrange(desc(Mean_Posterior))

# Step 3: Select specific rows
n <- nrow(posterior_summary)
selected_rows <- c(1, 250, 350, 498:503,n - 349, n - 249, n)  # 100th, 90th, 48–53, 10th lowest, lowest

# Step 4: Extract those rows
selected_samples <- posterior_summary[selected_rows, ]

# View selected samples
print(selected_samples)

#now to plot

library(ggplot2)
library(dplyr)

# 1. Filter posterior_df for selected IDs
selected_ids <- selected_samples$ID

# Ensure the order matches selected_samples$ID
posterior_filtered <- posterior_df %>%
  filter(ID %in% selected_samples$ID) %>%
  mutate(ID = factor(ID, levels = selected_samples$ID))

# 2. Plot histogram faceted by ID
ggplot(posterior_filtered, aes(x = Predicted_Prob)) +
  geom_histogram(binwidth = 0.02, color = "black") +
  facet_wrap(~ ID, scales = "free_y") +
  labs(title = "Posterior Probability Distributions over 1000 Iterations",
       x = "Predicted Posterior Probability",
       y = "Frequency") +
  theme_minimal(base_size = 14)

####Now we take the 6 most frequent predictors and build the model with them.

library(MASS)      # for LDA
library(dplyr)     # for data manipulation

# 1. Get the top 6 most frequent predictors
top_predictors <- predictor_counts %>%
  arrange(desc(Count)) %>%
  slice_head(n = 6) %>%
  pull(Predictor)

# 2. Prepare the dataset
# Make sure df has Y as the outcome column and predictor columns named as in top_predictors
df_lda2 <- df %>%
  dplyr::select(Y, all_of(top_predictors)) %>%
  mutate(Y = as.factor(Y))  # LDA needs a factor outcome

# 3. Fit LDA model
lda_fit2 <- lda(Y ~ ., data = df_lda2)

# 4. Predict posterior probabilities
lda_pred2 <- predict(lda_fit2)

# 5. Add the posterior probability of class 1 to the original dataframe
df_pred <- df %>%
  mutate(Posterior_Prob = lda_pred2$posterior[, "1"])

# View the result
head(df_pred)

####Now explore threshold cut offs on sensitivity and speficity.

# Convert Y to numeric if it's a factor
df_pred$Y <- as.numeric(as.character(df_pred$Y))

# Define a sequence of thresholds
thresholds <- seq(0, 1, by = 0.01)

# Initialize results dataframe
threshold_results <- data.frame(
  Threshold = thresholds,
  Sensitivity = NA,
  Specificity = NA
)

# Loop through each threshold and calculate sensitivity & specificity
for (i in seq_along(thresholds)) {
  t <- thresholds[i]
  
  predicted_class <- ifelse(df_pred$Posterior_Prob >= t, 1, 0)
  
  TP <- sum(predicted_class == 1 & df_pred$Y == 1)
  TN <- sum(predicted_class == 0 & df_pred$Y == 0)
  FP <- sum(predicted_class == 1 & df_pred$Y == 0)
  FN <- sum(predicted_class == 0 & df_pred$Y == 1)
  
  sensitivity <- ifelse((TP + FN) > 0, TP / (TP + FN), NA)
  specificity <- ifelse((TN + FP) > 0, TN / (TN + FP), NA)
  
  threshold_results$Sensitivity[i] <- sensitivity
  threshold_results$Specificity[i] <- specificity
}

threshold_results$FNR <-1- threshold_results$Specificity

library(ggplot2)

ggplot(threshold_results, aes(x = FNR, y = Sensitivity)) +
  geom_line(color = "darkgreen", size = 1) +
  labs(
    title = "Sensitivity vs False Negative Rate (FNR)",
    x = "False Negative Rate (FNR)",
    y = "Sensitivity"
  )

library(ggplot2)
library(tidyr)
library(dplyr)

# Convert to long format for ggplot
threshold_long <- threshold_results %>%
  pivot_longer(cols = c(Sensitivity, Specificity),
               names_to = "Metric",
               values_to = "Value")

# Plot both Sensitivity and Specificity vs. Threshold
ggplot(threshold_long, aes(x = Threshold, y = Value, color = Metric)) +
  geom_line(size = 1) +
  scale_color_manual(values = c("Sensitivity" = "blue", "Specificity" = "red")) +
  labs(
    title = "Sensitivity and Specificity vs Threshold",
    x = "Threshold",
    y = "Metric Value",
    color = "Metric"
  )

#####Bonus testing on dummy novel dataset#####

set.seed(456)  # Different seed for test set

# Selected predictors
selected_preds <- c("P256", "P292", "P94", "P143", "P81", "P244")

# Extract original predictor columns from df
orig_data <- df[, selected_preds]

# Number of test observations
n_test <- 50

# Define outcome with minority positives (e.g., 15 positives, 35 negatives)
n_pos <- 15
n_neg <- n_test - n_pos
Y_test <- c(rep(1, n_pos), rep(0, n_neg))

# Initialize test matrix
X_test <- matrix(NA, nrow = n_test, ncol = length(selected_preds))
colnames(X_test) <- selected_preds

# For each selected predictor, generate test data mimicking the original distributions by group
for (pred in selected_preds) {
  # Split original data by Y
  pos_vals <- orig_data[df$Y == 1, pred]
  neg_vals <- orig_data[df$Y == 0, pred]
  
  # Estimate means and sds for pos and neg groups
  mean_pos <- mean(pos_vals)
  sd_pos <- sd(pos_vals)
  mean_neg <- mean(neg_vals)
  sd_neg <- sd(neg_vals)
  
  # Generate test values separately for pos and neg observations
  X_test[Y_test == 1, pred] <- rnorm(n_pos, mean = mean_pos, sd = sd_pos)
  X_test[Y_test == 0, pred] <- rnorm(n_neg, mean = mean_neg, sd = sd_neg)
}

# Combine into a data frame
test_df <- data.frame(ID = paste0("Test", 1:n_test),
                      Y = Y_test,
                      X_test)

# Inspect test data
head(test_df)
dim(test_df)

# Ensure that Y is factor for prediction dataset as well
test_df$Y <- as.factor(test_df$Y)

# Use only the predictors used in lda_fit2 for prediction
# Assuming lda_fit2 was trained on the predictors in 'top_predictors'
predictors_used <- lda_fit2$terms %>% 
  attr("term.labels")  # Extract predictors from model terms

# Predict posterior probabilities on test dataset
lda_test_pred <- predict(lda_fit2, newdata = test_df[, predictors_used])

# Add posterior probabilities for class "1" to test_df
test_df_pred <- test_df %>%
  mutate(
    Posterior_Prob = lda_test_pred$posterior[, "1"],
    Posterior_Class = lda_test_pred$class  # No subsetting needed here
  )

# View first few rows with predictions
head(test_df_pred)

#assess the prediction ability. 
# Ensure both columns are factors with the same levels
test_df_pred <- test_df_pred %>%
  mutate(
    Y = as.factor(Y),
    Posterior_Class = as.factor(Posterior_Class)
  )

# Load the caret package
# install.packages("caret")  # Uncomment if not already installed
library(caret)

# Generate confusion matrix
cm <- confusionMatrix(test_df_pred$Posterior_Class, test_df_pred$Y, positive = "1")

# Print the full confusion matrix with metrics
print(cm)

