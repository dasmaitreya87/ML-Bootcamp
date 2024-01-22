<a name="_2gk3dm8jutmz"></a>**Machine Learning Bootcamp** 

<a name="_xgoo6gfnxofj"></a>**Project Report**

Maitreya Das

<23je0536@iitism.ac.in>
# <a name="_8v8vt3bihiuj"></a>Contents



|**Serial No**|**Content Name**|**Page No**|
| :-: | :-: | :- |
|1|Introduction|1|
|2|Linear Regression|1-3|
|3|Polynomial Regression|3-4|
|4|Logistic Regression|4-6|
|5|K Nearest Neighbors|6-7|
|6|N Layer neural networks|7-8|
|7|K Means Clustering|8-10|
|8|Conclusion|10|

# <a name="_itandb5cmzx4"></a>Introduction
In the rapidly evolving landscape of technology, the intersection of artificial intelligence and machine learning has become a focal point for innovation and advancement. Recognizing the significance of staying abreast of these transformative technologies, I embarked on a journey of intensive learning through a comprehensive machine learning bootcamp.

Six key machine learning algorithms are covered in detail in this project: N-Layer Neural Networks, K-Nearest Neighbors, K-means Clustering, Polynomial Regression, and Linear Regression.
# <a name="_uq3hl2d0hdlz"></a>Linear Regression
Project Link :[Project_Linear_Regression_Gradient_Descent](https://colab.research.google.com/drive/1eINVWi_X4-iB2YNJhqTdx9AEguc1GOpG?usp=drive_link)

The various components of the code are hereby discussed
### <a name="_ybibsfkrxrrg"></a>Normalize Features
Np.mean : calculates the mean of each column in the input matrix X along axis = 0.

Np.std : Calculates the standard deviation of each column in the input matrix X along axis = 0.

This process ensures that each feature has zero mean and unit variance (standard deviation of 1)
### <a name="_p49xalnsj89x"></a>Computing Cost
Predictions : are calculated by taking the dot product of the input feature matrix X and the parameter vector theta.

Mean Squared Error : is calculated by taking the sum of squared differences between the predicted values and actual target values and dividing by 2\*m (m=len(y))
### <a name="_159zi4mbc73a"></a>Gradient Descent Function
For obtaining the optimized value of the parameters,theta.

Learning rate : The step size used in updating the parameters during each iteration of gradient descent.

Cost history : list to store the cost at each iteration for later analysis or plotting.

### <a name="_2k4lcoofwsuq"></a>R-2 Score
Ss\_total : computes the total sum of the squares, which represents the total variance. It calculates the squared differences between each true value and the mean of the true values.

Ss\_residual : Computes the residual sum of squares, which represents the unwxplained variance between the true target value and predicted values.

### <a name="_y8zvka6vdf6k"></a>Adding the bias term
The bias term is commonly added to the features in linear regression to account for a constant offset or intercept in the model. This is often represented as an additional feature with a constant value of 1.
### <a name="_9h776ogd346k"></a>Generating the weights
Np.random.randn : This generates random values from a standard normal distribution. The number of random values generated is equal to the number of columns in X\_normalized\_b.

The random weights are updated during the training process using the gradient descent algorithm.
### <a name="_vma5bvnb9ej"></a>Graph of cost vs iterations

R-squared score: 0.9999999978525291
# <a name="_hiv1z3f0imdx"></a>Polynomial Regression
Project Link :[Project_Polynomial_Regression_Gradient_Descent](https://colab.research.google.com/drive/1N82XLLJTEz_OCZT7XN9-VIKniTVbMH2j?usp=drive_link)

Polynomial regression is a specialized form of linear regression. The main difference lies in the preprocessing step for polynomial regression, where **additional polynomial features** are created. Once this step is completed, the coding structure for training and making predictions remains highly similar to that of linear regression.
### <a name="_vhzjyey13wly"></a>Generating Polynomial Features
X\_poly = np.ones((m, 1)) 

Initializes a feature matrix that includes the bias term only.

The following powers are included : 

Individual powers

X1^2,X2^3,X3^4

Cross Terms

X1\*X2,X1\*X3,X2\*X3

Mixed Powers

X1^2\*X2,X1\*X2^2,X2^3\*X3
### <a name="_kehoza50t1rk"></a>Multivariable Polynomial Regression Function
The parameters included in this part of the program are : 

1) Degree : degree of the polynomial
1) Alpha : Learning rate
1) Num\_iterations : Number of iterations for gradient descent
1) Theta : Coefficients of the polynomial regression model
### <a name="_funxl6rg8vcz"></a>The Hyperparameters
alpha=0.0001 #Learning rate

num\_iterations=1000

theta=multiple\_variable\_polynomial\_regresssion(X\_poly,y\_train,degree,alpha,num\_iterations)
### <a name="_97acubvc7tnl"></a>R2 Score Values
For the train data : 

R2 Score: 0.814010224089485

For the test data : 

R2 Score: 0.8052817171547013

# <a name="_q07sxr7wwhzo"></a>Logistic Regression
Project Link : [Project_Logistic_Regression](https://colab.research.google.com/drive/17JcAsAaCWyLb601qMfj69IO92CDO0Ihv?usp=sharing)
### <a name="_t5lnqa6c3hj9"></a>One-hot encoding technique

It is a common technique used in machine learning to represent categorical labels in a binary matrix format.

Each row corresponds to a sample, and each column corresponds to a class.

Encoded\_labels : initializes a matrix with zeros. Each row of this matrix will represent a sample, and each column will represent a class.
### <a name="_qjtr9nqx9lg"></a>Sigmoid Function

A sigmoid function is employed to squash the output of a linear transformation into a range between 0 and 1.

The logistic regression model computes a linear combination of input features, and the sigmoid transforms it into a probability.
### <a name="_44omh87lymjq"></a>Initialize parameters function
W : The weight matrix has dimensions (num\_features,num\_classes); each column represents weights associated with one class, and each row corresponds to a feature.

B : The bias term.
### <a name="_nlnuk1qhm5u3"></a>Forward Propagation
Z=np.dot(X,W)+b

Computes the linear transformation of the input features X using the weights W and the bias vector b. Calculates the weighted sum of the input features.

The sigmoid function squashes the output to a range between 0 and 1, representing the probability of belonging to the positive class.

A=sigmoid(Z)
### <a name="_4l3yw517umgx"></a>Compute cost function
Cross entropy loss between predicted probabilities A and true labels Y. This loss is generally calculated in cases of binary classification. The same loss will be calculated in case of n-layer neural networks later on.

The formula for the same is: 

Numerical Stability

**np.log(A+1e-15)**

A small epsilon is added to the predicted probabilities before taking the logarithm. This is done to avoid issues with taking the logarithm of zero.

cost=-1/m\*np.sum(Y\*np.log(A+1e-15)) #Add small epsilon to avoid log(0)
### <a name="_i63ut03aenhi"></a>Backward Propagation Formula
Computing the gradients of the cost function with respect to the parameters (Weights and bias)

dZ : This is the derivative of the cost with respect to pre-activation Z. In the case of binary classification, it is simply the difference between predicted probabilities A and actual labels.

dW : Stores the gradient of the cost function with respect to the weights.

db : Stores the gradient of the cost function with respect to the bias.
### <a name="_nrykc86t1d6j"></a>Training function
This function trains the training dataset by calling all the various functions. It also prints the cost every 100 iterations.
### <a name="_4c2q93kiegl5"></a>Prediction function
Using the forward propagation function defined earlier, it gets the numpy array A.

**np.argmax(A,axis=1) :** This function returns the index of the maximum value in each row. For each row of the matrix A, it finds the index (column number) where the maximum value occurs.

**Note : 
The assumption here is that the logistic regression model is designed for a binary classification task where there are two classes (0 and 1).**
### <a name="_smz5c55hfgho"></a>Assessment of the model
Precision : 0.9396303715903693

Recall : 0.9353333333333333

F1 score : 0.9341149059462862
# <a name="_gmucc1ogmz84"></a>K-Nearest Neighbors
Project Link : [KNN_Trial_WORKING](https://colab.research.google.com/drive/1BE-DZxY70b4Aw8r7YB2SMdUaQ6M9hS4d?usp=drive_link)
### <a name="_wsmzutx0057r"></a>Standard Scaling
The mathematical formula for standard scaling is 

This ensures that each feature has a mean of 0 and a standard deviation of 1. Both the training and testing data are scaled to ensure consistency.
### <a name="_lndon2vd80h"></a>Euclidean Distance
The mathematical function for calculating euclidean distance is

### <a name="_r8w8eobfdzgp"></a>K\_nearest\_neighbors function
The **enumerate** function is a built-in Python function that takes an iterable (in this case, X\_test\_scaled) and returns an iterator that produces tuples containing the index and the corresponding value from the iterable.

For each test instance, distances to all training instances are computed using Euclidean distance (euclidean\_distance is assumed to be a function that calculates Euclidean distance).

**Np.argsort** : This NumPy function returns the indices of the array elements in a sorted manner. Also here, the first k indices are only considered (the k nearest neighbors).

The labels corresponding to the indices are extracted and stored in k\_neighbors\_labels.

**Np.bincount** :  is a NumPy function that counts occurrences of non-negative integers in an array. The resulting array contains the counts of occurrences for each unique non-negative integer in the input array. The index i in the resulting array corresponds to the count of occurrences of the integer i in the original array.

**Np.argmax** : is a NumPy function that returns the index of the maximum value along a specified axis of the array.
### <a name="_3p4wragooo9j"></a>Optional Printing of cost
After every 100 iterations, the code prints the iteration number and the cost, which is calculated as **1 - accuracy.** The accuracy is the proportion of correct predictions up to the current iteration. Formula of accuracy : 

### <a name="_sv617hpoah11"></a>Assessing the K-nearest neighbors model

Precision :0.9622176948391702

Recall : 0.9618333333333333

F1 score : 0.961811315318202
# <a name="_ifon79v5sgub"></a>N-Layer Neural Networks
Project Link : [N_Layer_Neural_Network_WORKING](https://colab.research.google.com/drive/1awTtUDYM6c-cmCAUORiG2Tl8lR2OdI_N?usp=sharing)

#Neural Network parameters

input\_size=num\_features

hidden\_size=128

output\_size=num\_classes

learning\_rate=0.001

epochs=1000
### <a name="_x2t174sek8sl"></a>Initializing the weights and biases
For the hidden layer : 

**Weights\_hidden** : Initializes the weights for the connections between the input layer and the hidden layer. The shape of the weight matrix is (input\_size,hidden\_size)

**Biases\_hidden** : Initializes the biases for the neurons in the hidden layer. Shape is (1,hidden\_size). There is one bias for each neuron in the hidden layer.

For the output layer : 

**Weights\_output** : Weights for the connection between the hidden layer and the output layer. Shape is (hidden\_layer,output\_size)

**Bias\_ouput** : biases for the neuron in output layer size are (1, output\_size). There is one bias for each neuron in the output layer.
### <a name="_eohtc2i64rjx"></a>Forward Pass
In the forward pass, the input data X\_train is multiplied by the weights of the hidden layer (weights\_hidden), and biases (biases\_hidden) are added. The result is passed through the ReLU activation function, and the output is calculated. Then, the output of the hidden layer is multiplied by the weights of the output layer (weights\_output), and biases (biases\_output) are added. Finally, the log-softmax activation function is applied to get the log probabilities.
### <a name="_3526d4dcpdg9"></a>Compute Loss
The cross-entropy loss is computed using the log probabilities (log\_probs) and the ground truth labels (y\_train). This is a measure of how well the model is performing on the training data.
### <a name="_xvgc4yk8o764"></a>Backward Pass
The gradients are computed for both the output layer (delta\_output) and the hidden layer (delta\_hidden). These gradients are used to update the weights and biases in the next step.
### <a name="_heb0di59lepc"></a>Update Weights and bias
The weights and biases are updated using gradient descent. This step involves subtracting a fraction of the gradient of the loss with respect to the parameters (weights and biases) from the current values. The learning\_rate parameter controls the size of the steps taken during optimization.
### <a name="_6bud9boqdnec"></a>Print Cost
For every 100 iterations, the code will print the loss.
### <a name="_gokh6qj8zumh"></a>Accuracy
Comparing the true\_labels\_test and the predicted\_labels\_test, the accuracy on the test dataset comes out to be 93.30%.

<a name="_47ofwrzggel6"></a>K-Means Clustering

Project Link : [K_Means_Clustering](https://colab.research.google.com/drive/1XiqU4nQY0HaWcyvoZmqzOoyt-loyZEqk?usp=sharing)
### <a name="_swgi84wibigt"></a>The function kmeans
centroids = X[np.random.choice(X.shape[0], k, replace=False)]

This line initializes ‘k’ centroids randomly by selecting ‘k’ unique data points from ‘X’.
### <a name="_btqr7j5d2ar8"></a>Assigning data points to clusters
distances = np.linalg.norm(X[:, np.newaxis] - centroids, axis=2)

This calculates the sum along the third axis, which effectively sums the squared differences for each data point and centroid pair.

labels = np.argmin(distances, axis=1)

Np.argmin is used to find the index of the minimum value along axis 1 in the distance matrix, where there is effective assignment of each datapoint to the closest centroid.
### <a name="_4idk56bbiv1h"></a>Updating the centroids
The algorithm updates the centroids by calculating the mean of all data points.

X[labels == j]. mean(axis=0)

In this line of code, firstly, this is a boolean indexing operation that selects all rows from the original dataset X. Then it calculates the mean along axis 0, which corresponds to taking the mean across all the selected data points for each feature.
### <a name="_wtaozcw5l58t"></a>Convergence Check
if np.all(centroids == new\_centroids):

`                `break

This part of the code is checking whether the centroids have converged, and if they have, then it breaks out of the loop, indicating the k means algorithm has reached a stable configuration.

The np.all() function returns true if all elements in the boolean array are true.
### <a name="_v06ykrxzh5a1"></a>Calculating the SSE
SSE stands for the sum of squared distances between the data points and the centroid assigned to that cluster. The main objective of kmeans clustering algorithm is to reduce the value of SSE, indicating that datapoints are getting closer to their respective centroids.
### <a name="_lviyulwnz4f2"></a>Elbow method function
The function iterates over different values of k, runs the kmeans function for each k, and collects the corresponding SSE values in the sse\_values list. Then, it plots the SSE values against the number of clusters using Matplotlib.
### <a name="_3dum5sp6y7gz"></a>My SSE vs k graph

We can see that the elbow of the graph is located at k = 3.The "elbow" in the plot is often considered the optimal value for k, where adding more clusters does not significantly reduce the SSE.

### <a name="_b7om2i9ibqeo"></a>Silhouette score
The silhouette score measures how well separated the clusters are and how similar an object is to its own cluster.The silhouette score for the entire dataset is the average of the silhouette score for each instance. The score ranges from -1 to 1, where a high value indicates that the object is well matched to its own cluster and poorly matched to neighboring clusters.

Silhouette score : 0.571138193786884

# <a name="_b57rqbazlso8"></a>Conclusion
In summary, I built all the algorithms for this ML bootcamp project from scratch, aiming to make them as accurate as possible. I learned from sources like YouTube, GeeksforGeeks, and Kaggle to gather information. By using different measures, I tried to ensure the algorithms worked well. This project reflects my commitment to learning about machine learning and the math behind it.



