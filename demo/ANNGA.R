#ANN demo
#Francis Roy-Desrosiers 2011


require(ANN)


demo.ANNGA <- function() {
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

cat("Next step should take less than 10 minutes\n")
readline("Press <Enter> to continue")

#maxGen  =10
set.seed(123)
res10           <-ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
                mutation = 0.3,
                crossover = 0.7,
                maxGen  =10,
                error   =0.001)

set.seed(123)
#maxGen  =100
res100          <-ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
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
                population  =100,
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

demo.ANNGA()
