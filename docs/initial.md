# Goals for this project 

1. Learn how these deep learning techniques actually work from soup to nuts. All of my work so far has been reading and understanding the math on these concepts. But there is a huge difference between actually understanding those concepts and just reading about them.  
This paper  
/home/nicole/Documents/mycorrhizae/efficentzerofromscratch/docs/papers/efficentzerov2  
is a state of the art reinforcement learning architecture designed to do similar stuff to alphago but with way fewer samples. Its the main idea for 


2. EVERYTHING MUST COME FROM SCRATCH. No dependencies except for extreme basics, the following must be forbidden.  
- No transformer or tensor libraries. ie candle / burn or other stuff  
- Dependencies for automatic differentiation.  Autograd baad.

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
(f(g(x)))' = f'(g(x)) * g'(x)
$$


Ok, so every layer can be represented as a vector of values.  And the state vector for layer $n+1$ can be represented as a function $f_{n}(V_{n}) = V_{n+1}$. Where this function for simple networks as multiplying a matrix representing the weights  

$$
f_n(V_n)= \sigma(W_n * (V_n) + B_n)
$$


So the formula for the final layer $V_n$ would look like  

$$
V_n = f_{n-1}\bigl(f_{n-2}(\dots f_2(f_1(V_0))\bigr) = \mathrm{NN}(V_0)
$$

So lets go ahead and try to split $f$ into 2 components.  

$$
\begin{aligned}
Z_k &= W_k \cdot V_{\,k-1} + B_k, \\
V_{\,k} &= \sigma(Z_k).
\end{aligned}
$$

Then this is going to get plugged into some kind of evaluator function $L(V_n)= \text{loss}$. And we want to find the gradient that decreases said loss. Which should throw out a partial $\displaystyle \frac{\partial L}{\partial V_n}$ that represents how to change the weights in the final layer to minimize the loss.

From here we need to do a couple things. We need to use the partial $\displaystyle \frac{\partial L}{\partial V_{\,k}}$ to figure out a couple things. Namely you need to figure out the partial  $\displaystyle \frac{\partial L}{\partial V_{\,k-1}}$ to continue the back propagation. You also need to figure out how to modify the weights in $f_k$ using the partial $\displaystyle \frac{\partial L}{\partial V_{\,k}}$. 

First part should be pretty easy through applying the chain rule once would be:

$$
\frac{\partial L}{\partial Z_k} 
= \frac{\partial L}{\partial V_{\,k}} \;\odot\; \sigma'(Z_k).
$$

The second part should look like this hopefully?

$$
\frac{\partial L}{\partial V_{\,k-1}} 
= W_{\,k}^{T} \; \frac{\partial L}{\partial Z_{\,k}}.
$$

Since for a hidden layer $k$, we have $\delta_k = \partial L/\partial Z_k$, we use

$$
\frac{\partial L}{\partial V_{\,k-1}} 
= W_{\,k}^{T}\,\delta_{\,k}.
$$

Then we define

$$
\delta_{\,k-1} \;=\; \frac{\partial L}{\partial Z_{\,k-1}} 
\;=\; \Bigl(\frac{\partial L}{\partial V_{\,k-1}}\Bigr)\;\odot\;\sigma'(Z_{\,k-1}).
$$

Whoo, so now how can I get the calculus for changing the weights figured out, biases are easy, since they linearly affect the activations. So taking the partial with respect to the biases just nukes the weights

$$
\frac{\partial L}{\partial B_k} = \frac{\partial L}{\partial Z_k}.
$$

Figuring out the weights seems a bit harder, but lets try to break it down in terms of a single weight value $(Z_k)_j$. Which by definition is going to be 

$$
(Z_k)_j = (B_k)_j + \sum_{i} (W_k)_{j,i} \,(V_{\,k-1})_{i}.
$$

Which means that if I take the partial with respect to a single weight value on layer $k$ with indices $j,i$, then

$$
\frac{\partial (Z_k)_j}{\partial (W_k)_{j,i}} = (V_{\,k-1})_{i}.
$$

By the chain rule,

$$
\begin{aligned}
\left(\frac{\partial L}{\partial Z_k}\right)_j 
= \sum_{r,s} \left(\frac{\partial L}{\partial W_k}\right)_{r,s}
\frac{\partial (W_{k,r,s})}{\partial (W_{k,j,i})}
\frac{\partial (Z_k)_j}{\partial (W_{k,j,i})}
= \left(\frac{\partial L}{\partial W_k}\right)_{j,i} \,(V_{\,k-1})_{i}.
\end{aligned}
$$

Therefore,

$$
\left(\frac{\partial L}{\partial W_k}\right)_{j,i} 
= \left(\frac{\partial L}{\partial Z_k}\right)_j \,(V_{\,k-1})_{i}.
$$
