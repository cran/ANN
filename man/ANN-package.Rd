\name{ANN-package}
\alias{ANN-package}
\docType{package}
\title{
Artificial Neural Network optimized by Genetic Algorithm  
}
\description{
This package provides a feedforward Artificial Neural Network (ANN) 
	optimized by Genetic Algorithm (GA). Modifications in the
	Genetic Algorithm from CUDAANN r6 have been done to 
	improve the speed, the convergence and the memory 
	management.
}
\details{
\tabular{ll}{
Package: \tab ANN\cr
Type: \tab Package\cr
Version: \tab 0.1 Beta\cr
Date: \tab 2011-08-22\cr
License: \tab GPL-3 \cr
LazyLoad: \tab yes\cr

}


An artificial neural network is  based on the mathematical representation of a neural network, in other words, the way our brain works. The artificial neural network (ANN) has neurons and synapses, similar to the neural network.  The synapses are the connections between the neurons which are represented as weight (\eqn{w_i}). There are several connection types which  exist to solve different kinds of problems. In this package, we use the  feedforward artificial neural network. In this type of network, the information is gathered into the input layer which forwards the information through the network to the output layer.   


The ANN consists of different layers. The input layer takes the input, then distributes it to the connections that connect the hidden layer(s) and the input layer. The neuron(s) in the hidden layer(s) process(es) the summation of the information received from the connections of the input layer. Then it processes the summations with its activation function and distributes the result to the next layer. This process continues down through the layers to the output layer. The neuron(s) of the output layer process(es) the summation of the information received from the connections of the hidden layer. Then each neuron processes the summation with its activation function. The output of the activation function is the output of the ANN. 


In this package, the activation function is sigmoid: \eqn{f(x) = \frac{1}{1+e^{-x}}}. An example of a formula for a 3 layer feedforward artificial neural network (the input layer, one hidden layer and the output layer) with one output would be:





\eqn{f(\sum_{j=1}^{N_{hidden \; layer}} w_{2,j,1}f(\sum_{i=1}^{N_{input \; layer}} w_{1,i,j} input_i + bias_{1,j})+bias_{2,1})  =output}




\eqn{w_{k,i,j}}, means weight from the \eqn{i}th neuron on the \eqn{k}th layer to the \eqn{j}th neuron on the next layer. The \eqn{bias_{k,j}} is similar to a constant in a least square regression. Each neuron of the hidden layer and of the output layer has a bias and the neurons add the bias to the summation.  The weights of the connections must be estimated. The learning phase of the ANN can be               supervised  or  unsupervised. In this package the supervised learning has been implemented. Supervised learning consists of giving inputs to the ANN and adjusting the weight to minimize the  sum of the differences between the predicted output given by the ANN and the desired output.  The mean squared error \eqn{ MSE =\frac{1}{N} \sum(y^{output}_i - y^{estimated \:\;  output}_i)^2} is the criterion to be minimized  in this package. 




Several methods of minimisation are available. In this package, the  Genetic Algorithm is used to solve the problem of estimating the connection's weights. Genetic Algorithm is based on natural evolution.  A population of chromosomes is initialised with random numbers. In our case a chromosome is an array of doubles corresponding to weights. This population of chromosomes evolves within every generation; mutation and crossover happen to all chromosomes.  Mutation ( \eqn{w^{new}_{j,i}=w_{k,i}+mutationRate*(w_{n,j}-w_{m,i}) }) consists of adding a weight \eqn{i} from a random chromosome to the difference of the weight \eqn{i} of two other random chromosomes multiplied by the mutation rate, where \eqn{w_{k,i}} is the \eqn{i}th weight of the \eqn{k}th chromosome and \eqn{k}, \eqn{n}, \eqn{m} are randomly choosen within the set of chromosomes population and \eqn{k}, \eqn{n}, \eqn{m} are not equal. The crossover consists of changing each chromosome's weights to the  mutation weight with the probability of the crossover rate. Then, it replaces the old chromosome with the new one if the MSE of the new chromosome is smaller. This algorithm of minimisation was taken from "CUDAANN r6 project developed by Emre Caglar". Two variants of this algorithm are implemented in this package and as described below.

}
\author{
Francis Roy-Desrosiers %, based on CUDAANN r6 project written by Emre Caglar.

Maintainer: Francis Roy-Desrosiers <francis.roy.desrosiers at gmail.com>

}

\note{
In order to use this package with multiple cores, the user must open the "Makevars" in the source code and follow the instructions. 
Then you will need to compile the C++ code,  open a terminal, and type:

cd "/path..." 

where "/path..." is the path to the source code. Then compile the code. 

R CMD INSTALL "nameOfTheFileWhereTheSourceCodeIs"

The "Makevars" has been built for Linux, unfortunatly  for Windows users. If anybody wants to make a "Makevars.win" for Windows or  develop this package, I will be more than happy to accept your contribution and add your name as a contributor. 
}
\references{
Cross Ss, Harrison Rf, Kennedy RL.. Introduction to neural networks. Lancet 1995; 346: 1075-1079. 


}
%%~~ Optionally other standard keywords, one per line, from file KEYWORDS in ~~
%%~~ the R documentation directory ~~
\keyword{ neural }
\keyword{ network }
\keyword{ genetic }


