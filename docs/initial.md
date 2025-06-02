# Goals for this project 

1. Learn how these deep learning techniques actually work from soup to nuts. All of my work so far has been reading and understanding the math on these concepts. But there is a huge difference between actually understanding those concepts and just reading about them.
This paper 
/home/nicole/Documents/mycorrhizae/efficentzerofromscratch/docs/papers/efficentzerov2
is a state of the art reinforcement learning architecture designed to do similar stuff to alphago but with way fewer samples. Its the main idea for 


2. EVERYTHING MUST COME FROM SCRATH. No dependancies except for extreme basics, the following must be forbidden.
- No transformer or tensor libraries. ie candle or other stuff
- Dependancies for automatic differentiation.  Autograd baad.

3. Rust for everything!! Largely just because of the type safety and ability to write very generic low level code. 

Also Rust-GPU has the potential to actually move a lot of this code to the GPU at some point. this is far far far outside the scope for this initial project.



# Architecture for the beginning component.

To get something working really really quickly, writing the code to test and train a simple reinforcement learning algorithm that can recognize the handwritten mnist dataset. Using the most basic reinforcement style learning stuff. 

Ok so I looked it up and the images are 28x28 pixels, with maybe a byte of brightness information per pixel. 

trying to just do everything from memory that means that the initial layer will need 784 nodes. 

I have no clue what I am doing but lets do 3 hidden layers with 20 nodes each. 

Then 10 layers representing the output with a node lighting up depending on the digit detected. 


So the equation for every single node's value is going to be something a 

Nonlinearity(Sum(previous_value_i * weight_i for i in previous layer) + bias)


So chain rule is:
$$
(f(g(x)))' = f'(g(x))*g'(x)
$$


Ok, so every layer can be represented as a vector of values.  And the state vector for layer $n+1$ can be represented as a function $f_{n}(V_{n}) = V_{n+1}$. Where this function for simple networks as multiplying a matrix representing the weights 

$$
f_n(V_n)= \sigma(W_n * (V_n) + B_n)
$$


So the formula for the final layer $V_n$ would look like 


$$
V(n)=f_{n-1}(f_{n-2}(\dots f_2(f_1(V_0)))= NN(V_0)
$$

Then this is going to get plugged into some kind of evaluator function $L(V_n)= loss$. And we want to find the gradient that decreases said loss. Which should throw out a partial $\partial V_n$ that represents how to change the weights in the final layer to minimize the loss.





So lets say that instead of the result of the final layer, 

