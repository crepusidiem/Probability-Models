# Probability-Models
This repository implements various probability models with real-world data.

## Milestone 2 Q&A
1. Explain what your AI agent does in terms of PEAS. What is the "world" like?

Our agent's Performance Measure is log-likelihood, given that we are building a Naive-Bayes model. Our agent's Environment includes everything our sensors are able to pick up and more. Other factors include other significant financial obligations on the customers' minds, the customers' portfolio and plans, as well as emergencies. Our agent's Actuators predict the probability of the final class label for a previously unseen customer. Finally, our agent's Sensors contain up to 17 features such as age, occupation, marital status, and more that allow the agent to learn patterns and make predictions.

2. What kind of agent is it? Goal based? Utility based? etc.

Our agent will be utility based, as it will seek to optimize a performance metric, as mentioned previously.

3. Describe how your agent is set up and where it fits in probabilistic modeling.

Our agent will utilize a Naive-Bayes model to make its predictions. It will optimize itself based on a log-likelihood, and the results it outputs will be our final prediction.
