\name{ANNGA}
\alias{ANNGA}
\alias{ANNGA.default}
\title{Optimize an ANN by GA}
\description{
  This function uses a feedforward Artificial Neural Network 
	optimized by Genetic Algorithm to minimize the mean squared error. 
}

\usage{
ANNGA	(	x,
		y,
		design = c(1, 3, 1),
		population = 500,
		mutation = 0.3,
		crossover = 0.7,
		maxW = 25,
		minW = -25,
		maxGen = 1000,
		error = 0.05,
		printBestChromosone=TRUE,
		...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{x}{
     A matrix of inputs, where each column is a different input.
	Value must be in the range [0,1]. 
}
  \item{y}{
     A matrix of outputs, where each column is a different output.
	Value must be in the range [0,1]. 
}
  \item{design}{
     A vector for the configuration of the network: the first 
	number will be the number of inputs, the last number will be the number of outputs and the numbers in between are the numbers of neurons on each hidden layer.

	Example 1:design = c(1, 3, 1) one input, one hidden layer
	with three neurons and one neuron in the output layer.

	Example 2: design = c(5, 3, 2, 1) five inputs, two hidden layers, the first one with three neurons, the second one with two neurons
	 and one neuron in the output layer.
}
  \item{population}{
     The number of chromosomes in the population of each generation.
}
  \item{mutation}{
     The mutation rate: should be between 0 and 1.
}
  \item{crossover}{
     The crossover rate: must be between 0 and 1.
}
  \item{maxW}{
     The maximum weight allowed by the random generator 
	on the initialisation of the weight. During the
	mutation, weight can excede this boundry.
}
  \item{minW}{
     The minimum weight allowed by the random generator 
	on the initialisation of the weight. During the
	mutation, weight can excede this boundry.
}
  \item{maxGen}{
     The maximum generation allowed if the error criterion is not reached.
}
  \item{error}{
    Error criterion to be reached. Desired mean squared error. 
}

  \item{printBestChromosone}{
	For a better understanding, this option prints the weight of the best chromosome
	for each generation. 
}
%   \item{threads}{
%	Number of threads to be used, if the number is greater than your CPU number of threads, then your number will be set to your CPU number of threads.
%}
  \item{\dots}{
    
}

}
%\details{
%}

%\value{
%%  ~Describe the value returned
% If it is a LIST, use
%%  \item{comp1 }{Description of 'comp1'}
%%  \item{comp2 }{Description of 'comp2'}
%% ...
%}
%\references{
%put references to the literature/web site here 
%}
\author{
Francis Roy-Desrosiesrs
}
\note{
Once an object of class "ANN" has been created, you can \code{print()} and \code{plot()} to visualise information about that object.


}

%% ~Make other sections like Warning with \section{Warning }{....} ~

%\seealso{
%% ~~objects to See Also as \code{\link{help}}, ~~~
%}
\examples{


#use the command demo(ANN) to see how dataANN was created 
data("dataANN")

#Example
	        ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
                mutation = 0.3,
                crossover = 0.7,
                maxGen  =1000,
                error   =0.001)


#More examples 
demo(ANN)
}
\keyword{ neural }
\keyword{ genetic }
