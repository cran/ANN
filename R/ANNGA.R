####################################################################
## ANNGA.R: Artificial Neural Network optimized by Genetic Algorithm 
##
## Copyright (C) 2011 Francis Roy-Desrosiers
##
## This file is part of ANN.
##
## ANN is free software: you can redistribute it and/or modify it
## under the terms of the GNU General Public License as published by
## the Free Software Foundation,  version 3 of the License.
##
## ANN is distributed in the hope that it will be useful, but
## WITHOUT ANY WARRANTY; without even the implied warranty of
## MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
## GNU General Public License for more details.
##
## You should have received a copy of the GNU General Public License
## along with ANN.  If not, see <http:##www.gnu.org/licenses/>.
####################################################################

ANNGA <-
function(x, ...)UseMethod("ANNGA")


ANNGA.default <-
function(x,
 	y, 
	design=c(1,3,1),
	maxPop=500,
	mutation=0.3,
	crossover=0.7,
	maxW=25,
	minW=-25,
	maxGen=1000,
	error=0.05,
	iMaxGenerationSameResult=100,
	bMaxGenerationSameResult=TRUE,
	bCycleGauss=FALSE,
	nbCycleGauss=1,
	nbCycleGaussBest=0,
	sigma=1,
	probGauss=0.5,
	printBestChromosone=TRUE,...) {

	input <- as.matrix(x)
	output <- as.matrix(y)
	if(any(is.na(x))) stop("missing values in 'x'")
	if(any(is.na(y))) stop("missing values in 'y'")
	if(dim(x)[1L] != dim(y)[1L]) stop("nrows of 'y' and 'x' must match")
	if(dim(input)[1L] <=0) stop("nrows of 'x' and 'y' must be >0 ")
	if (maxPop<20){
		cat("The population should be over 20,maxPop=",maxPop, "  , the default population is 200   \n")
		maxPop<-200
	}
	if(sigma<=0) stop("'sigma' must be positive")
	   
	est <- .Call("ANNGA" ,              
                x,
		y, 
		design,
		maxPop,
		mutation,
		crossover,
		maxW,
		minW,
		maxGen,
		error,
		iMaxGenerationSameResult,
		bMaxGenerationSameResult,
		bCycleGauss,
		nbCycleGauss,
		nbCycleGaussBest,
		sigma,
		probGauss,
		printBestChromosone,
                PACKAGE="ANN")

	if(dim(output)[2]==1){est$R2<-1-sum((output-est$output)^2)/sum((output-mean(output))^2)}else{est$R2<-NULL}
	est$call <- match.call()
	class(est) <- "ANN"
	#print(est)
	est
}




print.ANN <-
function(x,...)
{
	cat("Call:\n")
	print(x$call)

	cat("\n****************************************************************************")
	cat("\nMean Squared Error------------------------------>",x$mse)
	if (!(is.null(x$R2))){
	cat("\nR2---------------------------------------------->",x$R2) 
	}else{cat("\nIf more than 1 output, R2 is not computed")}
	cat("\nNumber of generation---------------------------->",x$nbOfGen)
	cat("\nWeight range allowed----------------------------> [",x$maxW,",",x$minW,"]")
	cat("\nWeight range resulted from the optimisation-----> [",max(x$bestChromosome),",",min(x$bestChromosome),"] ")
	cat("\nMutation rate started at", x$startMutation ," and finished at",x$endMutation)
	cat("\n****************************************************************************\n\n")

}


predict.ANN <-
function(object,input,...)
{
	if (is.null(input)) stop("'input' is missing", call. = FALSE)
	if(any(is.na(input))) stop("missing values in 'input'")
	if(class(object)!="ANN") stop("object must be a ANN class ")
	if(dim(input)[1L] <=0) stop("nrows of 'input' must be >0 ")
	input <- as.matrix(input)
	est <- .Call("predictANNGA",               # either new or classic
		         input, object$nbNeuronPerLayer,object$bestChromosome,
		         PACKAGE="ANN")
	est
}


plot.ANN <-
function(x,...)
{
	if(dim(x$desiredOutput)[2L]>1) stop("output must be univariate")
	#par(mfrow=c(1,1))
	plot(x$desiredOutput,xlab="x axis", ylab="y axis")
	lines(x$output,col="red")
	title("Neural Network output vs desired output")
	legend("topleft", c("desired Ouput","Output"), cex=0.6, bty="n", fill=c("black","red"))
}

