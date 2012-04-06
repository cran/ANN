////////////////////////////////////////////////////////////////////
// RcppANNGA.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
// Copyright (C) 2011-2012 Francis Roy-Desrosiers
//
// This file is part of ANN.
//
// ANN is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, either version 3 of the License, or
// (at your option) any later version.
//
// ANN is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ANN.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////


#include <cmath>
#include "files/Training.cpp"
#include <R.h>
#include <Rinternals.h>
#include <Rmath.h>

extern "C"{

////////////////////////////////////////////////////////
SEXP ANNGA(SEXP matrixInput,
	SEXP matrixOutput,
	SEXP design, 
	SEXP maxPop, 
	SEXP mutation, 
	SEXP crossover, 
	SEXP maxW, 
	SEXP minW, 
	SEXP maxGen, 
	SEXP error,
	SEXP rprintBestChromosome,
	SEXP rthreads,
	SEXP rCppSeed) {


////////////////////////////////////////////////////////
///////////////////INITIALIZATION///////////////////////
////////////////////////////////////////////////////////
	
	
	int *Rdim = INTEGER(getAttrib(matrixOutput, R_DimSymbol));
	int kOut = Rdim[1];
	Rdim = INTEGER(getAttrib(matrixInput, R_DimSymbol));
	int kIn = Rdim[1];	  //cols
	int lengthData = Rdim[0]; //rows
		
	double** matIn = new double*[lengthData];
	double** matOut = new double*[lengthData];

	/* EXAMPLE R->CPP matrix
	for( i=0; i<nrow; i++){ 
                for( j=0; j<ncol; j++){ 
                        Rprintf( " %f ", p[i+nrow*j] ) ; 
                } 
                Rprintf( "\\n" ) ; 
        };*/ 

	for (int i=0; i<lengthData; i++) {
	matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = REAL(matrixInput)[i+lengthData*j];
	    	}
	}
	for (int i=0; i<lengthData; i++) {
		matOut[i]=new double[kOut];
	    	for (int j=0; j<kOut; j++) {
			matOut[i][j] = REAL(matrixOutput)[i+lengthData*j];
	    	}
	}

	int *nbNeuronPerLayer;
	nbNeuronPerLayer=INTEGER(design);
	int nbOfLayer = length(design);

	int 	iMaxPop		=INTEGER(maxPop)[0];
	double 	dmutation 	=REAL(mutation)[0];
	double 	dcrossover	=REAL(crossover)[0];
	double 	dmaxW		=REAL(maxW)[0];
	double 	dminW		=REAL(minW)[0];
	int 	imaxGen		=INTEGER(maxGen)[0];
	double 	derror		=REAL(error)[0];
	bool	printBestChromosome = (bool)INTEGER(rprintBestChromosome)[0];
	int	threads 	= INTEGER(rthreads)[0];
	int	cppSeed		= INTEGER(rCppSeed)[0];


////////////////////////////////////////////////////////
////////////////////NETWORK CREATION////////////////////
////////////////////////////////////////////////////////


	ANNTraining *ANNT= new ANNTraining(nbOfLayer , nbNeuronPerLayer ,lengthData, matIn, matOut, iMaxPop, dmutation,  dcrossover, dminW, dmaxW, printBestChromosome, threads, cppSeed);


////////////////////////////////////////////////////////
///////////////////NETWORK OPTIMIZATION/////////////////
////////////////////////////////////////////////////////


	ANNT->initializePopulation ();


	while(ANNT->mGenerationNumber<= imaxGen && ANNT->mFitnessValues[ANNT->bestIndividual]>derror){
		ANNT->cycle (true); 
	}

	ANNT->getANNresult(true);


////////////////////////////////////////////////////////
//////////////////OUTPUT FOR R//////////////////////////
////////////////////////////////////////////////////////


	SEXP  list, list_names;
	PROTECT(list = allocVector(VECSXP, 16));    
	PROTECT(list_names = allocVector(VECSXP,16));

	SEXP output, chromosome, mse, nbOfGen, RvectorFitness;

	SET_VECTOR_ELT(list_names,0,mkChar("input"));
	SET_VECTOR_ELT(list, 0, matrixInput);

	SET_VECTOR_ELT(list_names,1,mkChar("desiredOutput"));
	SET_VECTOR_ELT(list, 1, matrixOutput);


	PROTECT(output = allocMatrix(REALSXP, lengthData, kOut));
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		REAL(output)[i+lengthData*j] = ANNT->outputANN[i][j];
	    }
	}
	SET_VECTOR_ELT(list_names,2,mkChar("output"));
	SET_VECTOR_ELT(list, 2, output);

	SET_VECTOR_ELT(list_names,3,mkChar("nbNeuronPerLayer"));
	SET_VECTOR_ELT(list, 3, design);

	SET_VECTOR_ELT(list_names,4,mkChar("maxPop"));
	SET_VECTOR_ELT(list, 4, maxPop);

	SET_VECTOR_ELT(list_names,5,mkChar("startMutation"));
	SET_VECTOR_ELT(list, 5, mutation);

	/*PROTECT(dendmutation 	= allocVector(REALSXP,1 ));
	REAL(dendmutation)[0]	=ANNT->crossRate;			
	SET_VECTOR_ELT(list_names,6,mkChar("endMutation"));
	SET_VECTOR_ELT(list, 6, dendmutation);*/

	SET_VECTOR_ELT(list_names,6,mkChar("crossover"));
	SET_VECTOR_ELT(list, 6, crossover);

	SET_VECTOR_ELT(list_names,7,mkChar("maxW"));
	SET_VECTOR_ELT(list, 7, maxW);

	SET_VECTOR_ELT(list_names,8,mkChar("minW"));
	SET_VECTOR_ELT(list, 8, minW);

	SET_VECTOR_ELT(list_names,9,mkChar("maxGen"));
	SET_VECTOR_ELT(list, 9, maxGen);
	
	SET_VECTOR_ELT(list_names,10,mkChar("desiredEror"));
	SET_VECTOR_ELT(list, 10, error);

	PROTECT(chromosome = allocVector(REALSXP,ANNT->mWeightConNum ));
	for (int i=0; i<ANNT->mWeightConNum; i++) {
		REAL(chromosome)[i] = ANNT->mChromosomes[ANNT->bestIndividual][i];
	}
	SET_VECTOR_ELT(list_names,11,mkChar("bestChromosome"));
	SET_VECTOR_ELT(list, 11, chromosome);

	PROTECT(mse	 	= allocVector(REALSXP,1 ));
	REAL(mse)[0]		=ANNT->mFitnessValues[ANNT->bestIndividual];
	SET_VECTOR_ELT(list_names,12,mkChar("mse"));
	SET_VECTOR_ELT(list,12, mse);

	PROTECT(nbOfGen	 	= allocVector(INTSXP,1 ));
	INTEGER(nbOfGen)[0]	=ANNT->mGenerationNumber;
	SET_VECTOR_ELT(list_names,13,mkChar("nbOfGen"));
	SET_VECTOR_ELT(list, 13, nbOfGen);

	PROTECT(RvectorFitness = allocVector(REALSXP,(int)ANNT->vectorFitness.size() ));
	for (int i=0; i<(int)ANNT->vectorFitness.size(); i++) {
		REAL(RvectorFitness)[i] = ANNT->vectorFitness[i];	
	}
	SET_VECTOR_ELT(list_names,14,mkChar("vectorFitness"));
	SET_VECTOR_ELT(list, 14, RvectorFitness);

	setAttrib(list, R_NamesSymbol, list_names);
	ANNT->release (); //release the memory, TO DO, NOT WORKING
	UNPROTECT(7);

return list;    
}



////////////////////////////////////////////////////////
SEXP predictANNGA(SEXP matrixInput,SEXP design, SEXP chromosome) {


////////////////////////////////////////////////////////
///////////////////INITIALIZATION///////////////////////
////////////////////////////////////////////////////////
 

	int *Rdim = INTEGER(getAttrib(matrixInput, R_DimSymbol));
	int lengthData = Rdim[0]; //rows
	int kIn = Rdim[1];	  //cols

	double** matIn = new double*[lengthData];

	for (int i=0; i<lengthData; i++) {
	matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = REAL(matrixInput)[i+lengthData*j];
	    	}
	}

	int *nbNeuronPerLayer;
	nbNeuronPerLayer=INTEGER(design);
	int nbOfLayer = length(design);
	int kOut = nbNeuronPerLayer[nbOfLayer-1];

	double *cppChromosome = REAL(chromosome);


////////////////////////////////////////////////////////
////////////////////NETWORK CREATION////////////////////
////////////////////////////////////////////////////////


	ANNTraining *ANNT= new ANNTraining(nbOfLayer , nbNeuronPerLayer ,lengthData, matIn);
	ANNT->ann->loadWights (cppChromosome);
	ANNT->getANNresult(false);



////////////////////////////////////////////////////////
//////////////////OUTPUT FOR R//////////////////////////
////////////////////////////////////////////////////////


	SEXP  list, list_names,output;
	PROTECT(list = allocVector(VECSXP, 1));    
	PROTECT(list_names = allocVector(VECSXP,1));

	PROTECT(output = allocMatrix(REALSXP, lengthData, kOut));
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		REAL(output)[i+lengthData*j] = ANNT->outputANN[i][j];
	    }
	}
	SET_VECTOR_ELT(list_names,0,mkChar("predict"));
	SET_VECTOR_ELT(list, 0, output);

	setAttrib(list, R_NamesSymbol, list_names);
	//ANNT->release (); //release the memory, TO DO NOT WORKING
	UNPROTECT(3);
    return list;
}


}//end extern "C"

