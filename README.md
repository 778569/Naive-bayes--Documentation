# Naive-bayes--Documentation
Naive Bayes is a simple yet effective classification algorithm based on Bayes' theorem. It assumes independence between features, making it computationally efficient and suitable for large datasets.

# Bayes’ Theorem 

![image](https://github.com/778569/Naive-bayes--Documentation/assets/52319671/7a42f240-6508-430f-b007-da54f2edd456)

 # Naive Bayes Assumption & Simplified Equation:

Naive Bayes assumes conditional independence between features (attributes) given the class label. This means:

P(X1, X2, ..., Xn | Y) = P(X1 | Y) * P(X2 | Y) * ... * P(Xn | Y)

Using this assumption, the classification problem becomes finding the class Y with the highest posterior probability given a new data point X:

P(Y | X) = P(X1, X2, ..., Xn | Y) * P(Y) / P(X1, X2, ..., Xn)

Since P(X1, X2, ..., Xn) is constant for all classes, we can maximize the product of class-conditional probabilities (P(X1 | Y) * P(X2 | Y) * ...) and class prior probabilities (P(Y)) to find the most likely class.

# Equations for Different Feature Types:

Discrete Features: P(Xi | Y) is estimated using frequency counts or smoothing techniques (Laplace smoothing)
Continuous Features: Gaussian Naive Bayes assumes features follow a normal distribution. P(Xi | Y) is modeled using the mean and standard deviation of each feature for each class.
Overall Prediction:

argmax_Y { P(X1 | Y) * P(X2 | Y) * ... * P(Xn | Y) * P(Y) }

Remember, while straightforward, the "naive" assumption of independence between features can limit the model's accuracy in complex scenarios.
