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


#use the command demo(ANN) to see how dataANN was created 
data("dataANN")

par( mfrow=c(2,2) )
                plot(sin(t), main="sin")
                plot(noisy_sin, main="sin + 0.5 * rnorm")
                plot(output, main="sin + 0.5 * rnorm in [0,1]")
                plot(input, main="input should be in the range [0,1]")


demoResultANN

ANNObject<-demoResultANN[[3]]


ANNObject 	#or print(ANNObject)
plot(ANNObject)


p		<-predict( ANNObject, input )
p
plot(p$predict,main="Predicted Output")


}
\keyword{ neural }
\keyword{ network }
\keyword{ genetic }
