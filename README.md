# Advanced-Data-Analytics-and-Machine-Learning-Techniques-in-R
An advanced data analytics project implemented in R featuring data visualization, classification analysis, association rule mining (Apriori), text analytics with corpus and word clouds, and clustering techniques such as K-means to extract patterns and insights from structured and unstructured data.
# Install necessary packages
install.packages("caTools")
install.packages("arules")
install.packages("rpart")
install.packages("tm")
install.packages("wordcloud")
install.packages("e1071")
install.packages("caret")
install.packages("ggplot2")
install.packages("dplyr")
install.packages("factoextra")

# Load necessary packages
library(caTools)
library(arules)
library(rpart)
library(tm)
library(wordcloud)
library(e1071)
library(caret)
library(ggplot2)
library(dplyr)
library(factoextra)

# Load the data
student_data <- read.csv("C:/Users/vaish/OneDrive/Desktop/student-por.csv")

# Dealing with Missing Data
FixNull <- function(attribute) {
  ifelse(is.na(attribute), mean(attribute, na.rm = TRUE), attribute)
}
student_data$famsize <- FixNull(student_data$famsize)
student_data$Medu <- FixNull(student_data$Medu)

# Splitting the Model
set.seed(12345)
sp <- sample.split(student_data$sex, SplitRatio = 0.6)
trainingSet <- subset(student_data, sp == TRUE)
testSet <- subset(student_data, sp == FALSE)

# Data Visualization
# Box Plot: Study time by number of failures
ggplot(student_data, aes(x = factor(failures), y = studytime)) +
  geom_boxplot() +
  labs(title = "Box Plot of Study Time by Number of Failures", x = "Number of Failures", y = "Study Time")

# Histogram: Distribution of Free Time
ggplot(student_data, aes(x = freetime)) +
  geom_histogram(binwidth = 1, fill = "skyblue", color = "black") +
  labs(title = "Histogram of Free Time", x = "Free Time", y = "Frequency")

# Scatter Plot: Age vs Study Time
ggplot(student_data, aes(x = age, y = studytime)) +
  geom_point(color = "blue", alpha = 0.5) +
  labs(title = "Scatter Plot of Age vs Study Time", x = "Age", y = "Study Time")

# Bar Plot: Counts of Family Size
ggplot(student_data, aes(x = famsize)) +
  geom_bar(fill = "coral", color = "black") +
  labs(title = "Bar Plot of Family Size", x = "Family Size", y = "Count")

# Density Plot: Distribution of Age
ggplot(student_data, aes(x = age, fill = sex)) +
  geom_density(alpha = 0.5) +
  labs(title = "Density Plot of Age Distribution by Sex", x = "Age", y = "Density")

# Association Analysis using Apriori Algorithm
itemset <- as(trainingSet[, c(3, 14)], "transactions")

# Perform Apriori analysis
rules <- apriori(itemset, parameter = list(support = 0.1, confidence = 0.7))

# Decision Tree Classification
tree_model <- rpart(famsize ~ ., data = trainingSet)

# Make predictions on the test set
predictions <- predict(tree_model, testSet, type = "class")

# Confusion Matrix
conf_matrix <- confusionMatrix(predictions, testSet$famsize)
print(conf_matrix)

# Naive Bayes Classification
target_variable <- "famsize"
predictor_variables <- setdiff(names(trainingSet), target_variable)

# Train the Naive Bayes model
nb_model <- naiveBayes(as.formula(paste(target_variable, "~", paste(predictor_variables, collapse = " + "))), data = trainingSet)

# Make predictions on the test set
nb_predictions <- predict(nb_model, testSet, type = "class")

# Generate the confusion matrix
confusion_matrix <- confusionMatrix(nb_predictions, testSet$famsize)
print(confusion_matrix)

# Text Mining
corpus <- Corpus(VectorSource(student_data$Fjob))

# Perform text cleaning
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))

# Find the top 10 terms in the corpus
term_freq <- DocumentTermMatrix(corpus)
term_freq_matrix <- as.matrix(term_freq)
term_freq_df <- data.frame(word = colnames(term_freq_matrix), freq = colSums(term_freq_matrix))
top_10_terms <- head(term_freq_df[order(term_freq_df$freq, decreasing = TRUE), ], 10)

# Display a word cloud for top words
wordcloud(words = top_10_terms$word, freq = top_10_terms$freq, min.freq = 1, scale = c(3, 0.2), colors = brewer.pal(8, "Dark2"))

# Confusion Matrix for Decision Tree
conf_matrix <- confusionMatrix(predictions, testSet$famsize)
print(conf_matrix)

# Convert predictions and testSet$famsize to factors with the same levels
predictions <- factor(predictions, levels = levels(testSet$famsize))

# Generate the confusion matrix for Decision Tree
conf_matrix <- confusionMatrix(predictions, testSet$famsize)
print(conf_matrix)

# Text Mining
corpus <- VCorpus(VectorSource(student_data$Fjob))

# Perform text cleaning
corpus <- tm_map(corpus, content_transformer(tolower))
corpus <- tm_map(corpus, removePunctuation)
corpus <- tm_map(corpus, removeNumbers)
corpus <- tm_map(corpus, removeWords, stopwords("en"))

# Find the top 10 terms in the corpus
term_freq <- DocumentTermMatrix(corpus)
term_freq_matrix <- as.matrix(term_freq)
term_freq_df <- data.frame(word = colnames(term_freq_matrix), freq = colSums(term_freq_matrix))
top_10_terms <- head(term_freq_df[order(term_freq_df$freq, decreasing = TRUE), ], 10)

# Display a word cloud for top words
wordcloud(words = top_10_terms$word, freq = top_10_terms$freq, min.freq = 1, scale = c(3, 0.2), colors = brewer.pal(8, "Dark2"))

# Plot word frequencies for top words
barplot(top_10_terms$freq, names.arg = top_10_terms$word, col = "skyblue", main = "Top 10 Words Frequency", xlab = "Word", ylab = "Frequency")

# Clustering Analysis
# Preprocess the data for clustering
clustering_data <- student_data %>%
  select(age, studytime, freetime, absences, G1, G2, G3) %>%
  scale()

# Determine the optimal number of clusters using the Elbow method
fviz_nbclust(clustering_data, kmeans, method = "wss")

# Perform k-means clustering with the chosen number of clusters (e.g., 3)
set.seed(123)
k <- 3
kmeans_result <- kmeans(clustering_data, centers = k, nstart = 25)

# Add the cluster assignments to the original dataset
student_data$cluster <- as.factor(kmeans_result$cluster)

# Visualize the clustering results
# Scatter plot of Age vs Study Time colored by clusters
ggplot(student_data, aes(x = age, y = studytime, color = cluster)) +
  geom_point() +
  labs(title = "K-means Clustering: Age vs Study Time", x = "Age", y = "Study Time")

# Scatter plot of G1 vs G3 colored by clusters
ggplot(student_data, aes(x = G1, y = G3, color = cluster)) +
  geom_point() +
  labs(title = "K-means Clustering: G1 vs G3", x = "G1 Grade", y = "G3 Grade")

# Summary of clusters
summary(kmeans_result)
