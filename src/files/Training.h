////////////////////////////////////////////////////////////////////
// Training.h: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANN project
// Copyright (C) 2011 Francis Roy-Desrosiers
//
// This file is part of ANN.
//
// ANN is free software: you can redistribute it and/or modify it
// under the terms of the GNU General Public License as published by
// the Free Software Foundation, version 3 of the License.
//
// ANN is distributed in the hope that it will be useful, but
// WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with ANN.  If not, see <http://www.gnu.org/licenses/>.
////////////////////////////////////////////////////////////////////

#ifndef ANNTRAINING_H
#define ANNTRAINING_H


#include "ANN.cpp"
#include <vector>

typedef double* Chromosome;



class ANNTraining{

public:	// to train the network
	ANNTraining(int nbLayers , int * neuronPerLayer ,int lengthData, double **tmatIn, double **tmatOut,  int iMaxPopulation, double dmutRate, double dcrossRate ,double dminW, double dmaxW,int iMaxGenerationSameResult,bool bMaxGenerationSameResult,double passSigma, double passProbGauss, bool rprintBestChromosome, int passCores); 
//	ANNTraining(int nbLayers , int * neuronPerLayer ,int lengthData, double *tmatIn, double *tmatOut,  int iMaxPopulation, double dmutRate, double dcrossRate ,double dminW, double dmaxW,int iMaxGenerationSameResult,bool bMaxGenerationSameResult,double passSigma, double passProbGauss, bool rprintBestChromosome, int passCores); 	
	// to predict from a trained the network
	ANNTraining(int nbLayers , int * neuronPerLayer ,int lengthData, double **tmatIn); 
	
	~ANNTraining();

	void		initializePopulation();
	void		mutate(int);
	void		crossover(int);
	void		crossoverGauss(int v);
	void		crossoverGaussBest(int v);
	void		select(int);
	int 		MaxPopulation;
	double 		mutRate;
	double 		crossRate;	
	double 		minW;
	double 		maxW;	
	double 		meanFitness;
	double		sigma;
	double		probGauss;
	
	void		calculateAllFitnessOfPopulation();	
	void		release();
	void		cycle(bool);
	void		cycleGauss (bool);
	void		cycleGaussBest (bool);
	void		printFitness();
	void		printFitness(int);
	double		getMinFitness();
	double		getFitness(Chromosome individual);
	void		setANNweightsWithBestChromosome();
	void		getANNresult();
	void		predictANN();

	Chromosome	*mChromosomes;		//population  - solution space
	Chromosome      bestChromosome;
	Chromosome	worstChromosome;
	int		bestIndividual;
	int		worstIndividual;

	double		*mFitnessValues;
	int		mPopulationSize;
	float		mfMutationRate;
	float		rateModificationMutRate;
	float		mfCrossoverRate;
	int		mLayerNum;
	int		*mNeuronNum;
	int		mWeightConNum;    //number of weight connections between neurons
	int		mGenerationNumber;
	int		generationSameResult;
	double		lastGenerationBest;
	int 		maxGenerationSameResult;
	bool		boolMaxGenerationSameResult;
	bool		printBestChromosome;
	int 		num_of_threads;	
	
	ArtificialNeuralNetwork* ann;
	double* 	trainInput;
	double* 	desiredOutput;

	Chromosome 	diff;
	Chromosome  	crossedTrialIndividual;
	vector<double> vectorFitness;   

	double**	dataIn;
	double**	dataOut;
	double**	outputANN;
	int		nbOfData;
	int		nbOfInput;
	int		nbOfOutput;

};




#endif
