////////////////////////////////////////////////////////////////////
// Training.h: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANN project
// Copyright (C) 2011-2012 Francis Roy-Desrosiers
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
	ANNTraining(int nbLayers , int * neuronPerLayer ,int lengthData, double **tmatIn, double **tmatOut,  int iMaxPopulation, double dmutRate, double dcrossRate ,double dminW, double dmaxW,bool rprintBestChromosome, int passThreads, unsigned int seed); 	
	// to predict from a trained the network
	ANNTraining(int nbLayers , int * neuronPerLayer ,int lengthData, double **tmatIn); 
	
	~ANNTraining();

	void		initializePopulation();
	void		mutate(int);
	void		crossover(int);
	void		select(int);
	int 		MaxPopulation;
	double 		mutRate;
	double 		crossRate;	
	double 		minW;
	double 		maxW;	
	double 		meanFitness;
	
	void		statPop();	
	void		release();
	void		cycle(bool);
	void		printFitness(int);
	double		getFitness(Chromosome);
	void		getANNresult(bool);

	Chromosome	*mChromosomes;		//population  - solution space
	Chromosome      bestChromosome;
	int		bestIndividual;

	double		*mFitnessValues;
	int		mPopulationSize;
	int		mLayerNum;
	int		*mNeuronNum;
	int		mWeightConNum;    	//number of weight connections between neurons
	int		mGenerationNumber;
	bool		printBestChromosome;
	int 		num_of_threads;	
	
	ArtificialNeuralNetwork* ann;

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
