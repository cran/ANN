####################################################################
## ANNGA.R: Artificial Neural Network optimized by Genetic Algorithm 
##
## Copyright (C) 2011-2012 Francis Roy-Desrosiers
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
function(x,
 	y, 
	design=c(1,3,1),
	population=500,
	mutation=0.3,
	crossover=0.7,
	maxW=25,
	minW=-25,
	maxGen=1000,
	error=0.05,
	printBestChromosone=TRUE,...)UseMethod("ANNGA")


ANNGA.default <-
function(x,
 	y, 
	design=c(1,3,1),
	population=500,
	mutation=0.3,
	crossover=0.7,
	maxW=25,
	minW=-25,
	maxGen=1000,
	error=0.05,
	printBestChromosone=TRUE,...) {


	threads=1 #openMP, still in developpement.
	input <- as.matrix(x)
	output <- as.matrix(y)
	if(any(is.na(input))) stop("ERROR: missing values in 'x'")
	if(any(is.na(output))) stop("ERROR: missing values in 'y'")
	if(dim(input)[1L] != dim(output)[1L]) stop("ERROR: nrows of 'y' and 'x' must match")
	if(dim(input)[1L] <=0) stop("ERROR: nrows of 'x' and 'y' must be >0 ")
	if (population<20){
		cat("The population should be over 20,maxPop=",population, "  , the default population is 100   \n")
		population<-100
	}
	if(threads<=0) stop("ERROR:  'threads' must be >0")

	cppSeed<-sample.int(2147483647, 1) 
	   
	est <- .Call('ANNGA' ,              
                input,
		output, 
		as.integer(design),
		as.integer(population),
		as.numeric(mutation),
		as.numeric(crossover),
		as.numeric(maxW),
		as.numeric(minW),
		as.integer(maxGen),
		as.numeric(error),
		as.integer(printBestChromosone),
		as.integer(threads),
		as.integer(cppSeed),
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
	cat("\nCall:\n")
	print(x$call)

	cat("\n****************************************************************************")
	cat("\nMean Squared Error------------------------------>",x$mse)
	if (!(is.null(x$R2))){
	cat("\nR2---------------------------------------------->",x$R2) 
	}else{cat("\nIf more than 1 output, R2 is not computed")}
	cat("\nNumber of generation---------------------------->",x$nbOfGen)
	cat("\nWeight range at initialization------------------> [",x$maxW,",",x$minW,"]")
	cat("\nWeight range resulted from the optimisation-----> [",max(x$bestChromosome),",",min(x$bestChromosome),"] ")
	cat("\n****************************************************************************")

	if(!(is.null(x$callpredict))){
		cat("\n\nCall predict:\n")
		print(x$callpredict)
		cat("\n*the result from predict() is in $predict ")
	cat("\n****************************************************************************")
	}
	cat("\n\n")
}


predict.ANN <-
function(object,input,...)
{
cat("*predict ANN object \n")
	input <- as.matrix(input)
	if (is.null(input)) stop("ERROR: 'input' is missing", call. = FALSE)
	if(any(is.na(input))) stop("ERROR: missing values in 'input'")
	if(class(object)!="ANN") stop("ERROR: object must be a ANN class ")
	if(dim(input)[1L] <=0) stop("ERROR: nrows of 'input' must be >0 ")
	input <- as.matrix(input)
	est <- .Call("predictANNGA",               
		         input, object$nbNeuronPerLayer,object$bestChromosome,
		         PACKAGE="ANN")
	est$callpredict <- match.call()
	est<-c(object,est)
	class(est) <- "ANN"
	est
}


plot.ANN <-
function(x,...)
{
	if(dim(x$desiredOutput)[2L]>1) stop("NOTE: to plot an ANN object output must be univariate")
	#par(mfrow=c(1,1))
	plot(x$desiredOutput,xlab="x axis", ylab="y axis")
	lines(x$output,col="red")
	title("Neural Network output vs desired output")
	legend("topleft", c("desired Ouput","Output"), cex=0.6, bty="n", fill=c("black","red"))
}

