////////////////////////////////////////////////////////////////////
// RcppANNGA.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
// Copyright (C) 2011 Francis Roy-Desrosiers
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

#include <RcppClassic.h>
#include <cmath>
#include "files/Training.cpp"




RcppExport SEXP ANNGA(SEXP matrixInput,
	SEXP matrixOutput,
	SEXP design, 
	SEXP maxPop, 
	SEXP mutation, 
	SEXP crossover, 
	SEXP maxW, 
	SEXP minW, 
	SEXP maxGen, 
	SEXP error,
	SEXP riMaxGenerationSameResult,
	SEXP rbMaxGenerationSameResult,
	SEXP rbCycleGauss,
	SEXP rnbCycleGauss,
	SEXP rnbCycleGaussBest,
	SEXP rsigma,
	SEXP rprobGauss,
	SEXP rprintBestChromosome) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {


	RcppMatrix<double> origInput(matrixInput);
	RcppMatrix<double> origOut(matrixOutput);
	int lengthData = origInput.rows(), kIn = origInput.cols(), kOut = origOut.cols();	
	double** matIn = new double*[lengthData];
	double** matOut = new double*[lengthData];


	for (int i=0; i<lengthData; i++) {
		matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = origInput(i,j);
	    	}
	}
	for (int i=0; i<lengthData; i++) {
		matOut[i]=new double[kOut];
	    	for (int j=0; j<kOut; j++) {
			matOut[i][j] = origOut(i,j);
	    	}
	}


	RcppVector<int> vecNeuronPerLayer(design);
	int nbOfLayer = vecNeuronPerLayer.size();
	int *nbNeuronPerLayer = new int[nbOfLayer];
	for (int i=0; i<nbOfLayer; i++) {
	    nbNeuronPerLayer[i] = vecNeuronPerLayer(i);
	}


	int 	iMaxPop		=Rcpp::as<int>(maxPop);
	double 	dmutation 	=Rcpp::as<double>(mutation);
	double 	dcrossover	=Rcpp::as<double>(crossover);
	double 	dmaxW		=Rcpp::as<double>(maxW);
	double 	dminW		=Rcpp::as<double>(minW);
	int 	imaxGen		=Rcpp::as<int>(maxGen);
	double 	derror		=Rcpp::as<double>(error);
	int 	iMaxGenerationSameResult= Rcpp::as<int>(riMaxGenerationSameResult);
	bool 	bMaxGenerationSameResult= Rcpp::as<bool>(rbMaxGenerationSameResult);
	int 	nbCycleGaussBest=Rcpp::as<int>(rnbCycleGaussBest);
	int 	nbCycleGauss	=Rcpp::as<int>(rnbCycleGauss);
	bool	bCycleGauss 	= Rcpp::as<bool>(rbCycleGauss);
	double 	dsigma		=Rcpp::as<double>(rsigma);	
	double 	dprobGauss	=Rcpp::as<double>(rprobGauss);	
	bool	printBestChromosome = Rcpp::as<bool>(rprintBestChromosome);

	ANNTraining *ANNT= new ANNTraining(nbOfLayer , nbNeuronPerLayer ,lengthData, matIn, matOut, iMaxPop, dmutation,  dcrossover, dminW, dmaxW, iMaxGenerationSameResult, bMaxGenerationSameResult, dsigma, dprobGauss, printBestChromosome);





	ANNT->initializePopulation ();
	ANNT->getMinFitness ();



	int kk=0;
	while(ANNT->mGenerationNumber< imaxGen && ANNT->mFitnessValues[ANNT->bestIndividual]>derror){
		ANNT->cycle (true); //Normal cycle
		kk++;		
		if(kk==nbCycleGauss){
			kk=0;
			if(bCycleGauss==true){
				ANNT->cycleGauss (true); // Cycle with gauss distribution around each Chromosomes
			}	
		}
	}
	kk=0;
	while(kk< nbCycleGaussBest && ANNT->mFitnessValues[ANNT->bestIndividual]>derror){
			kk++;
			ANNT->cycleGaussBest (true); // Cycle with gauss distribution around the best Chromosomes
	}



	ANNT->getANNresult();
	RcppMatrix<double> output(lengthData, kOut); 	// reserve n by k matrix
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		output(i,j) = ANNT->outputANN[i][j];
	    }
	}


	RcppVector<double> chromosome(ANNT->mWeightConNum);
	for (int i=0; i<ANNT->mWeightConNum; i++) {
		chromosome(i) = ANNT->mChromosomes[ANNT->bestIndividual][i];
	}

	double	mse = 	ANNT->mFitnessValues[ANNT->bestIndividual];
	int	nbOfGen=ANNT->mGenerationNumber;
	double  dendmutation=ANNT->mfMutationRate;

	RcppVector<double> rcppVectorFitness(ANNT->vectorFitness.size());
	for (int i=0; i<(int)ANNT->vectorFitness.size(); i++) {
		//rcppVectorFitness(i) = ANNT->vectorFitness(i);
		rcppVectorFitness(i) = ANNT->vectorFitness[i];	
	}
	

	RcppResultSet rs;


	rs.add("input", origInput);
	rs.add("desiredOutput", origOut);
	rs.add("output", output);
	rs.add("nbNeuronPerLayer", vecNeuronPerLayer);
	rs.add("MaxPop", iMaxPop);
	rs.add("startMutation", dmutation);
	rs.add("endMutation", dendmutation);
	rs.add("crossover", dcrossover);
	rs.add("maxW", dmaxW);
	rs.add("minW", dminW);
	rs.add("maxGen", imaxGen);
	rs.add("error", derror);
	rs.add("bestChromosome", chromosome);
	rs.add("mse", mse);  
	rs.add("nbOfGen", nbOfGen);  
	rs.add("vectorFitness", rcppVectorFitness);


	rl = rs.getReturnList();
	//ANNT->release (); //release the memory	


    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}




RcppExport SEXP predictANNGA(SEXP matrixInput,SEXP design, SEXP gene) {

    SEXP rl = R_NilValue; 		// Use this when there is nothing to be returned.
    char *exceptionMesg = NULL;

    try {

	RcppMatrix<double> origInput(matrixInput);
	int lengthData = origInput.rows(), kIn = origInput.cols();	
	double** matIn = new double*[lengthData];


	for (int i=0; i<lengthData; i++) {
		matIn[i]=new double[kIn];
	    	for (int j=0; j<kIn; j++) {
			matIn[i][j] = origInput(i,j);
	    	}
	}

	RcppVector<int> vecNeuronPerLayer(design);
	int nbOfLayer = vecNeuronPerLayer.size();
	int *nbNeuronPerLayer = new int[nbOfLayer];
	for (int i=0; i<nbOfLayer; i++) {
	    nbNeuronPerLayer[i] = vecNeuronPerLayer(i);
	}
	int kOut = vecNeuronPerLayer(nbOfLayer-1);

	RcppVector<double> RcppGene(gene);
	int nbOfConnections = RcppGene.size();
	double *cppGene = new double[nbOfConnections];
	for (int i=0; i<nbOfConnections; i++) {
	    cppGene[i] = RcppGene(i);
	}
	

	ANNTraining *ANNT= new ANNTraining(nbOfLayer , nbNeuronPerLayer ,lengthData, matIn);
	ANNT->ann->loadWights (cppGene);
	ANNT->predictANN();

	RcppMatrix<double> output(lengthData, kOut);
	for (int i=0; i<lengthData; i++) {
	    for (int j=0; j<kOut; j++) {
		output(i,j) = ANNT->outputANN[i][j];
	    }
	}

	RcppResultSet rs;
	rs.add("output", output);

	// Get the list to be returned to R.
	rl = rs.getReturnList();
	//ANNT->release (); //release the memory
	
    } catch(std::exception& ex) {
	exceptionMesg = copyMessageToR(ex.what());
    } catch(...) {
	exceptionMesg = copyMessageToR("unknown reason");
    }
    
    if(exceptionMesg != NULL)
	Rf_error(exceptionMesg);

    return rl;
}


