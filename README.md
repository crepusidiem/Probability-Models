# Probability-Models
This repository implements various probability models with real-world data.

## Milestone 2 Q&A
1. Explain what your AI agent does in terms of PEAS. What is the "world" like?

Our agent's Performance Measure includes multiple performance metrics such as accuracy, precision, and recall. Our agent's Environment contains the historical records including the socioeconomic background, demographics, and contact details of the transaction. Our agent's Actuators predict the probability of the final class label for a previously unseen customer. Finally, our agent's Sensors contain up to 17 features such as age, occupation, marital status, and more that allow the agent to learn patterns and make predictions.

2. What kind of agent is it? Goal based? Utility based? etc.

Our agent will be utility based, as it will seek to optimize the performance metrics mentioned previously. It seeks to optimize itself so that it will produce the best possible accuracy, precision, and recall scores.

3. Describe how your agent is set up and where it fits in probabilistic modeling.

Our agent will utilize a Naive-Bayes model to make its predictions. It will optimize itself based on a differentiable loss function like cross-entropy, and the results it outputs will be our final prediction.
