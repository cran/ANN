\name{predict.ANN}
\alias{predict}
\alias{predict.ANN}
\title{Predict the output with an ANN already optimised.}
\description{Once the Artificial Neural Network is optimized, the output can be predicted.}

\usage{
\method{predict}{ANN}(object, input,...)
}
%- maybe also 'usage' for other objects documented here.
\arguments{
  \item{object}{Object of class "ANN", an already optimized ANN.}
  \item{input}{Input inserted into the ANN  to predict outputs.}
  \item{\dots}{not used}
}
%\details{
%  to be continued...
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
%\note{
%further notes
%}

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


data("dataANN")
demoResultANN
ANNObject<-demoResultANN[[3]]
#to compute demoResultANN use :demo(ANNGA)

par( mfrow=c(1,1) )
print(ANNObject)
plot(ANNObject)

predictRes      <-predict( ANNObject, input )
plot(predictRes$output,main="Predicted Output")


}
\keyword{ neural }
\keyword{ network }
\keyword{ genetic }
