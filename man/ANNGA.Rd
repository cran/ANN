\name{ANNGA}
\alias{ANNGA}
\alias{ANNGA.default}
\alias{plot.ANN}
\alias{print.ANN}
\title{Calling the C++ code to create an ANN and to optimize it by GA}
\description{
  This function uses a feedforward Artificial Neural Network 
	optimized by Genetic Algorithm to minimize the mean squared error. 
}

\usage{
ANNGA.default(	x,
		y,
		design = c(1, 3, 1),
		maxPop = 500,
		mutation = 0.3,
		crossover = 0.7,
		maxW = 25,
		minW = -25,
		maxGen = 1000,
		error = 0.05,
		iMaxGenerationSameResult = 100,
		bMaxGenerationSameResult = TRUE,
		bCycleGauss=FALSE,
		nbCycleGauss=1,
		nbCycleGaussBest=0,
		sigma=1,
		probGauss=0.5,
		printBestChromosone=TRUE,...)
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
  \item{maxPop}{
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
  \item{iMaxGenerationSameResult}{
     Number of generations before diminution of the 
	mutation rate. If the error does not become smaller 
	for the number of generations set in 
	\code{iMaxGenerationSameResult} then the mutation rate is
	 multiplied by 0.9. 
}
  \item{bMaxGenerationSameResult}{
     To allow the diminution of the mutation rate, set \code{bMaxGenerationSameResult=TRUE} otherwise \code{bMaxGenerationSameResult=FALSE}.
}
  \item{bCycleGauss}{
	To allow the refinement of each chromosome with Gaussian random number, set \code{bCycleGauss=TRUE} otherwise \code{bCycleGauss=FALSE}.
}
  \item{nbCycleGauss}{
	To refine each chromosome. It uses a GA with Gaussian random number 
	around each chromosome. To understand where and when this GA cycle happens, 
	see the code in the Details section. In order to use this option, \code{bCycleGauss} needs to be set to \code{TRUE}.
    
}
  \item{nbCycleGaussBest}{
	To refine the best chromosome. In order to refine the best  chromosome 
	set a possitive value. It uses a GA with Gaussian random number 
	around the best chromosome. This process happens after the normal GA cycle and the GA cycle with Gaussian random number around each chromosome. 	To understand where and when this GA cycle happens, see the Details section. 
}
  \item{sigma}{
	Sigma use by the random Gaussian number generator, if Gaussian algorithm is used.
}
  \item{probGauss}{
	Probability of adding, or not, a Gaussian random number, if Gaussian algorithm is used.
}


  \item{printBestChromosone}{
	For a better understanding, this option prints the weight of the best chromosome
	for each generation. 
}
  \item{\dots}{
    
}

}
\details{

This package offers two variants of the Genetic Algorithms. The first one refines the weights around the best chromosome. The second one refines the weights  around each chromosome.


I developed the algorithm to improve  the fitness of the  population of chromosomes   with weights derived from its best chromosomes. Eventualy, the GA will no longer improve the fitness during the evolution of the generations. One way of getting a better fitness is to create a new population from the weights of the best chromosome in combination with random numbers from the Gaussian distribution. This algorithm uses the \code{probGauss} as the probability of adding, or not, a Gaussian random number, with mean equal to zero and variance equal to the \code{sigma}, to the weights from the best chromosome.


The algorithm used to improve the fitness of the  population of chromosomes   with weights derived from the same population was created from a partnership with Maxime Leroux.
It uses the same logic with the Gaussian distribution as above but  derived from the entire population of chromosomes. It converges more slowly than the previous variant (around the best chromosome) but pursues the objective of finding the global minimum of the mean squared error. Once the original Genetic Algorithm and the Genetic Algorithm with Gaussian random number around each chromosome have optimized the weights, the Genetic Algorithm with Gaussian random number around the best chromosome can be used.

The Genetic Algorithm searches the global minimum of the MSE function; these two algorithms have been created to improve it. While the Genetic Algorithm of CUDAANN r6 reaches a point near the global minimum,  these two algorithms described above  might refine the weights to get closer to the global minimum. These two algorithms need to be used carefully because generalisation could be affected.
 
while(GenerationNumber< maxGen && FitnessValues[bestIndividual]>error)\{

	cycle(); //Normal cycle

	kk++;		

	if(kk==nbCycleGauss)\{

		kk=0;

		if(bCycleGauss==true)\{

				cycleGauss(); 

		\}
		
	\}
\}

kk=0;

while(kk< nbCycleGaussBest && mFitnessValues[bestIndividual]>error)\{

		kk++;

		cycleGaussBest(); // Cycle with Gauss distribution around the best Chromosome

\}

Explanation of the code: the program enters in a \code{while} loop and it will run until one of the two conditions happen to be \code{false}. The two conditions that need to be filled are: the fitness of the best chromosome should be smaller than the \code{error} chosen and the number of generations should be equal to the maximum number of generations chosen in \code{maxGen}. When these conditions are not filled, generations of normal cycles occur. The number set in \code{nbCycleGauss} provides the number of normal cycles before one gaussian cycle is processed. The gaussian cycles refine  all chromosomes with Gaussian random distribution number. 

Once one of the two conditions is filled  refinements around the best chromosone can be made. This can improve the fitness result but generalisation could be affected. The refinement around the best chromosome has two conditions that need to be filled: the fitness of the best chromosome should be smaller than the error chosen and the number of generations should be equal to the maximum number of generations chosen in \code{nbCycleGaussBest}.





}
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
require(ANN)
set.seed(123)
t               <-seq(0, 2 * pi, 0.01)
output1         <-matrix( sin(t) )
output2         <-matrix( sin(t) + 0.5 * rnorm( length(t) ) )

fctLogistic <-  function(input) 
{
        input   <-as.vector(input)
        logit   <- 1/(1 + exp( -input ))
        return(logit)
} 

output          <-matrix( fctLogistic(output2) )
input           <-matrix( seq( 0, 1, length.out=length(t) ) )
par( mfrow=c(2,2) )
                plot(output1, main="sin")
                plot(output2, main="sin + 0.5 * rnorm")
                plot(output, main="sin + 0.5 * rnorm in [0,1]")
                plot(input, main="input should be in the range [0,1]")


#maxGen  =10
set.seed(123)
res10           <-ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                maxPop  =100,
                mutation = 0.3,
                crossover = 0.7,
                maxGen  =10,
                error   =0.001)

set.seed(123)
#maxGen  =100
res100          <-ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                maxPop  =100,
                mutation = 0.3,
                crossover = 0.7,
                maxGen  =100,
                bMaxGenerationSameResult = FALSE,
                error   =0.001)

set.seed(123)
#maxGen  =1000
res1000         <-ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                maxPop  =100,
                mutation = 0.3,
                crossover = 0.7,
                maxGen  =1000,
                bMaxGenerationSameResult = FALSE,
                error   =0.001)


par( mfrow=c(2,2) )
plot(res10)
title(sub="10 generations")
plot(res100)
title(sub="100 generations")
plot(res1000)
title(sub="1000 generations")

}
\keyword{ neural }
\keyword{ genetic }
