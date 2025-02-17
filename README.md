# Bank Marketing: Probability Models (README.md Explanation)

## Data Exploration & Preprocessing Steps
Dataset we use: https://archive.ics.uci.edu/dataset/222/bank+marketing
Our project analyzes a bank marketing dataset to predict whether a client will subscribe to a term deposit (y). To achieve this, we preprocess the data in the following ways:
### 1. Handling Missing Values
First, we check for missing values in the dataset. Based on our analysis, there are no missing values in any columns. This ensures data completeness and prevents issues with training the model.
### 2. Identifying Continuous and Categorical Variables
We classify features into continuous and discrete (categorical) variables:
- Continuous Features: `age`, `balance`, `duration`, `campaign`, `pdays`, `previous`
- Categorical Features: All other non-numeric columns, such as `job`, `marital`, `education`, `default`, `housing`, `loan`, `contact`, `month`, `poutcome`
### 3. Feature Selection and Discretization
To improve model performance, we evaluate the uniqueness of values in continuous features. Since `balance` has a high number of distinct values (more than 5% of the dataset), we apply quantile-based binning to convert it into discrete bins, reducing its complexity while preserving meaningful differences.
### 4. One-Hot Encoding for Categorical Features
Categorical variables like `job`, `marital`, and `education` are transformed using one-hot encoding, converting them into binary columns. For instance, if the job has categories like `admin`, `technician`, and `blue-collar`, we create separate columns (`job_admin`, `job_technician`, `job_blue-collar`) where each row has a 1 in the corresponding job column.
### 5. Data Splitting
Finally, we split the dataset into training and testing sets using `train_test_split` to ensure our model generalizes well to unseen data.

This preprocessing pipeline ensures that our data is clean, structured, and suitable for training a Naive Bayes classifier.

## More Q&A
### 1. Explain what your AI agent does in terms of PEAS. What is the "world" like?
Our agent's Performance Measure is log-likelihood, given that we are building a Naive-Bayes model. Our agent's Environment includes everything our sensors can pick up and more. Other factors include other significant financial obligations on the customers' minds, the customers' portfolios, and plans, as well as emergencies. Our agent's Actuators predict the probability of the final class label for a previously unseen customer. Finally, our agents' sensors contain up to 17 features such as age, occupation, marital status, and more that allow the agent to learn patterns and make predictions.

### 2. What kind of agent is it? Goal-based? Utility-based? etc.
Our agent will be utility-based, as it will seek to optimize log-likelihood, as mentioned previously.

### 3. Describe how your agent is set up and where it fits in probabilistic modeling.
Our agent will utilize a Naive-Bayes model to make its predictions. Our reasoning is as follows: if we take the input feature "campaign" (number of calls during the contract period) as an example, we believe that the causation flows from the output (a successful deposit) to the input (multiple calls made to confirm details and obtain additional information), and not the other way around. Hence, we arrive at a graph structure resembling that of a Naive-Bayes model, where given the output, all of our inputs/features are conditionally independent (i.e. all our features are d-separated via the output). It will optimize itself based on a log-likelihood function, and the results it outputs will be our final prediction on whether or not the customer subscribed.
