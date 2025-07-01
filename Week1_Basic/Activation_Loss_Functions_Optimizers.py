'''
Activation Functions in Neural Networks
Activation functions introduce non-linearity to neural networks. Without them, a neural network (even a deep one) would behave just like a linear regression model.
•	Use ReLU in hidden layers for general-purpose MLPs and CNNs.
•	Use Tanh in RNNs/LSTMs.
•	Use Sigmoid or Softmax at the output layer based on the task.
•	Use GELU, Swish in modern Transformer-based models.


What is a Loss Function?
A loss function tells the model how wrong its prediction was.
It computes the error between:
Ground truth (true label) vs model prediction (ŷ) 
This error (loss) is then minimized using optimization techniques.


What is an Optimizer?
An optimizer adjusts the model’s weights after computing loss, using gradients from backpropagation.
💡 Goal: Reduce the loss by updating weights to minimize the error.
wt+1=wt−η⋅∇wL(w)
•	w: weights
•	η (eta): learning rate
•	L: loss function
•	∇wL: gradient of the loss wrt weights

Stochastic Gradient Descent (SGD):
Simple and powerful baseline optimizer.
w=w−η⋅∇L
Pros:
•	Simple and memory efficient
Cons:
•	Can get stuck in local minima
•	Sensitive to learning rate

Adam (Adaptive Moment Estimation)
Combines:
•	Momentum (moving average of gradients)
•	RMSprop (scaling gradients by average of squared grads)
θ=θ−η⋅vt+ϵmt
Where:
•	mt: First moment (mean of gradients)
•	vt: Second moment (variance of gradients)
Pros:
•	Works well for most problems
•	Faster convergence
•	Handles sparse gradients

Common combo:
•	Binary classification → Sigmoid + BCELoss + Adam
•	Multi-class classification → CrossEntropy + Adam
•	Regression → MSE + SGD or Adam


'''
