////////////////////////////////////////////////////////////////////
// Training.cpp: Artificial Neural Network optimized by Genetic Algorithm 
// Based on CUDAANN r6 project
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

#ifndef ANNTRAINING_CPP
#define ANNTRAINING_CPP
#include "Training.h"
#include "fstream"
#include "iostream"

#include <R.h>
#include <Rmath.h>
#include <vector>

#ifdef _OPENMP
#include <omp.h>
#endif



using namespace std;

inline  double unifRand(double min, double max)
{   
return (rand() * ((max - min) / (double)RAND_MAX)) + min;
}


inline int unifRandInt(int lower,int upper)
{
return (int)(rand() % (upper - lower + 1) + lower);
}




//Constructor for ANNGA.default
ANNTraining::ANNTraining(int nbLayers,int * neuronPerLayer,int lengthData,double **tmatIn,double **tmatOut,int iMaxPopulation,double dmutRate,double dcrossRate,double dminW,double dmaxW,  bool rprintBestChromosome, int passThreads, unsigned int seed){
	
			srand (seed); //SET SEED
			mPopulationSize		= iMaxPopulation;
			mLayerNum		= nbLayers;
			mGenerationNumber	= 1;
			MaxPopulation		=iMaxPopulation;
			mutRate			=dmutRate;
			crossRate		=dcrossRate;	
			minW			=dminW;
			maxW			=dmaxW;	
			meanFitness		=0;
			printBestChromosome 	=rprintBestChromosome;

			/*
			num_of_threads = passThreads;
			#ifdef _OPENMP
			omp_set_num_threads(num_of_threads);
			#endif
			*/

			ann = new ArtificialNeuralNetwork(nbLayers,neuronPerLayer);

			//save neuron numbers in each layer
			mNeuronNum = new int[nbLayers];
			for(int i = 0 ; i < nbLayers ; i++)
				mNeuronNum[i] = neuronPerLayer[i];

			//find the total connection number between neurons 
			 mWeightConNum = 0;			
			 for (int i = 1 ; i < nbLayers ; i++){
				 mWeightConNum +=neuronPerLayer[i] * neuronPerLayer[i-1] + mNeuronNum[i];
			 }

			diff = new double[mWeightConNum];
			crossedTrialIndividual = new double[mWeightConNum];

			//create fitness vector
			mFitnessValues = new double[mPopulationSize]; 
			//fitness i is the fitness value related to i th individual mChromosomes[i] 

			//create population vectors
			mChromosomes = new Chromosome[mPopulationSize];
			int ii;
			 
			for(ii = 0 ; ii < mPopulationSize ; ii++){
				mChromosomes[ii] = new double[mWeightConNum];
				mFitnessValues[ii] = 0; 
			}
			 
			nbOfData		= lengthData;
			nbOfInput		= neuronPerLayer[0];
			nbOfOutput		= neuronPerLayer[nbLayers-1];

			dataIn = tmatIn;
			dataOut = tmatOut;
			outputANN = new double*[nbOfData];

			for(ii = 0 ; ii < nbOfData ; ii++){
				outputANN[ii] = new double[nbOfOutput];
			}

			vectorFitness.clear();			
}
//Constructor for predict.ANN
ANNTraining::ANNTraining(int nbLayers,int * neuronPerLayer,int lengthData,double **tmatIn){
		
			mLayerNum		= nbLayers;
			
			//create neural network 
			ann = new ArtificialNeuralNetwork(nbLayers,neuronPerLayer);
			//save neuron numbers in each layer
			mNeuronNum = new int[nbLayers];
			for(int i = 0 ; i < nbLayers ; i++)
				mNeuronNum[i] = neuronPerLayer[i];

			//find the total connection number between neurons 
			 mWeightConNum = 0;			
			 for (int i = 1 ; i < nbLayers ; i++)
				 mWeightConNum +=neuronPerLayer[i] * neuronPerLayer[i-1] + mNeuronNum[i];

			nbOfData		= lengthData;
			nbOfInput		= neuronPerLayer[0];
			nbOfOutput		= neuronPerLayer[nbLayers-1];


			dataIn  = tmatIn;

			//allocate memory for output data - represented as 2D matrix - 
			outputANN = new double*[nbOfData];
			for(int i = 0 ; i < nbOfData ; i++)
				outputANN[i] = new double[nbOfOutput];

}
////////////////////////////////////////////
ANNTraining::~ANNTraining(){}
////////////////////////////////////////////
void ANNTraining::statPop (){ 
	
	meanFitness=0;
	double best 	= 1000000000.0;		
	int tempB 	= 0;
	int i;
	
	for(int i = 0 ; i < mPopulationSize ; i++){ 
		//mFitnessValues[i] =getFitness (mChromosomes[i]); 

		meanFitness+= mFitnessValues[i];

		if(mFitnessValues[i] < best){
			best = mFitnessValues[i];
			tempB = i;
		}
	}
	
	meanFitness=meanFitness/(double)mPopulationSize;
	bestIndividual  = tempB;
	bestChromosome=mChromosomes[bestIndividual];	                             
}
////////////////////////////////////////////
void  ANNTraining::initializePopulation(){

	int i;
	 
	for(i = 0 ; i < mPopulationSize ; i++){
		for(int j = 0 ; j < mWeightConNum ; j++){
			mChromosomes[i][j] = unifRand(minW, maxW);
		}
		mFitnessValues[i] =getFitness (mChromosomes[i]);
	}

	statPop ();
}
////////////////////////////////////////////
void	ANNTraining::mutate(int v){
	
	int r1,r2,r3;
	do{
		r1 = unifRandInt (0,mPopulationSize-1);
		r2 = unifRandInt (0,mPopulationSize-1);
		r3 = unifRandInt (0,mPopulationSize-1);
	}while(r1 == r2 || r1 == r3 || r2 == r3 || r1 == v || r2 == v || r3 == v);

	int i;
	
	for( i = 0 ; i < mWeightConNum ; i++){
		diff[i] = mChromosomes[r1][i] + (mutRate * (mChromosomes[r3][i] - mChromosomes[r2][i]));
	}
}
////////////////////////////////////////////
void	ANNTraining::crossover(int v){

	double ran;
	int i;
	
	for( i = 0 ; i < mWeightConNum ; i++){
		ran =	unifRand(0.0,1.0);
		if(ran < crossRate){
			crossedTrialIndividual[i] = diff[i];
		}else{
			crossedTrialIndividual[i] = mChromosomes[v][i];
		}
	}
}
////////////////////////////////////////////
void		ANNTraining::select(int v){

	double fitnessTrialIndividual = getFitness (crossedTrialIndividual);
	if(fitnessTrialIndividual <  mFitnessValues[v]){
		int i;
		 
		for( i = 0 ; i < mWeightConNum ; i++){
			mChromosomes[v][i] =crossedTrialIndividual[i];
		}
		mFitnessValues[v]=fitnessTrialIndividual;	
	}	
}
////////////////////////////////////////////					    
double		ANNTraining::getFitness(Chromosome individual){

	ann->loadWights (individual);
	double sumError = 0;
	for(int i = 0 ; i < nbOfData ; i++){ //for each training sample
		ann->feedForward (dataIn[i]);
		sumError += ann->getMeanSquareError (dataOut[i]);
	}	
	return sumError/(double)nbOfData;	
}
////////////////////////////////////////////
void ANNTraining::getANNresult(bool loadBest=true){

	if(loadBest) ann->loadWights (mChromosomes[bestIndividual]);

	for(int i = 0 ; i < nbOfData ; i++){  
		ann->feedForward (dataIn[i]);
		ann->getOutput(i);
		for(int j = 0 ; j < nbOfOutput ; j++){
			outputANN[i][j] = ann->getOutput(j);
		}
	}
}
////////////////////////////////////////////
void		ANNTraining::cycle (bool print = false){
	Rprintf ("\n***cycle***\n");
	for(int i = 0 ; i < mPopulationSize ; i++){
		mutate(i);
		crossover(i);
		select(i);
	}

	statPop();
	
	if(print==true){printFitness (bestIndividual);}

	++mGenerationNumber;				
	vectorFitness.push_back (mFitnessValues[bestIndividual]);
}
////////////////////////////////////////////
void ANNTraining::printFitness (int i){

	Rprintf("Generation:  %d  Best population fitness : %4.8f Mean of population:%4.8f  ", mGenerationNumber, mFitnessValues[i],meanFitness);

	if(printBestChromosome==true){	
		Rprintf("\n Best chromosome->");
		for(int j = 0 ; j < mWeightConNum ; j++){
				Rprintf("%4.2f/", mChromosomes[i][j]);
		}
	}
	Rprintf("\n");
}
////////////////////////////////////////////
void		ANNTraining::release (){	

	for (int i = 0; i < mPopulationSize; ++i)
		delete [] mChromosomes[i];
	delete [] mChromosomes;

	delete [] mFitnessValues;
	delete [] mNeuronNum; 
	delete [] diff;
	delete [] crossedTrialIndividual;
	delete [] dataIn;
	delete [] dataOut;

	for(int i = 0 ; i < nbOfData ; i++)
		delete [] outputANN[i];
	delete [] outputANN;

	ann->release ();
}
////////////////////////////////////////////




#endif
