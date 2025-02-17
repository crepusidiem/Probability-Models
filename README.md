# Probability-Models
This repository implements various probability models with real-world data.

## Milestone 2 Q&A
1. Explain what your AI agent does in terms of PEAS. What is the "world" like?

Our agent's Performance Measure is log-likelihood, given that we are building a Naive-Bayes model. Our agent's Environment includes everything our sensors are able to pick up and more. Other factors include other significant financial obligations on the customers' minds, the customers' portfolio and plans, as well as emergencies. Our agent's Actuators predict the probability of the final class label for a previously unseen customer. Finally, our agent's Sensors contain up to 17 features such as age, occupation, marital status, and more that allow the agent to learn patterns and make predictions.

2. What kind of agent is it? Goal based? Utility based? etc.

Our agent will be utility based, as it will seek to optimize a performance metric, as mentioned previously.

3. Describe how your agent is set up and where it fits in probabilistic modeling.

Our agent will utilize a Naive-Bayes model to make its predictions. Our reasoning is as follows: if we take the input feature "campaign" (number of calls during the contact period) as an example, we believe that the causation flows from the output (a successful deposit) to the input (multiple calls made to confirm details and obtain additional information), and not the other way around. Hence, we arrive at a graph structure resembling that of a Naive-Bayes model, where given the output, all of our inputs/features are conditionally independent (i.e. all our features are d-separated via the output). It will optimize itself based on a log-likelihood function, and the results it outputs will be our final prediction on whether or not the customer subscribed.
