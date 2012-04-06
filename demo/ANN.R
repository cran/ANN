#ANN demo
#Francis Roy-Desrosiers 2011-2012


require(ANN)


demo.ANN <- function() {
set.seed(1)
t               <-seq(0, 2 * pi, 0.01)
#sin(t) + noise
noisy_sin       <-sin(t) + 0.5 * rnorm( length(t) ) 

fLog            <-  function(x)	1/(1 + exp( -x ))


output          <-matrix( fLog(noisy_sin) )
input           <-matrix( seq( 0, 1, length.out=length(t) ) )

par( mfrow=c(2,2) )
                plot(sin(t), main="sin")
                plot(noisy_sin, main="sin + 0.5 * rnorm")
                plot(output, main="sin + 0.5 * rnorm in [0,1]")
                plot(input, main="input should be in the range [0,1]")


cat("\nNext step should take less than 2 minutes\n")
readline("Hit <Return> to continue")

#maxGen  =10
set.seed(1)
res10<-	        ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
                mutation = 0.2,
                crossover = 0.6,
		minW	=-10,
		maxW	=10,
                maxGen  =10,
                error   =0.001)

#maxGen  =100
set.seed(1)
res100<-	ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
                mutation = 0.2,
                crossover = 0.6,
		minW	=-10,
		maxW	=10,
                maxGen  =100,
                error   =0.001)

#maxGen  =1000
set.seed(1)
res1000<-	ANNGA(x    =input,
                y       =output,
                design  =c(1, 3, 1),
                population  =100,
                mutation = 0.2,
                crossover = 0.6,
		minW	=-10,
		maxW	=10,           
		maxGen  =1000,
                error   =0.001)


par( mfrow=c(2,2) )
plot(res10)
title(sub="10 generations")
plot(res100)
title(sub="100 generations")
plot(res1000)
title(sub="1000 generations")





}

demo.ANN()
