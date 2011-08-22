////////////////////////////////////////////////////////////////////
// Training.cpp: Artificial Neural Network optimized by Genetic Algorithm 
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

#ifndef ANNTraining_H
#define ANNTraining_H
#include "Training.h"
#include "fstream"
#include "iostream"
#include <omp.h> // Include OpenMP
#include <R.h>
#include <Rmath.h>
#include <vector>

using namespace std;

inline  double unifRand(double min, double max)
{   
GetRNGstate();
double final =unif_rand() * (max - min ) + min;
 PutRNGstate();
return final;

}


int unifRandInt(int lower,int upper)
{
GetRNGstate();
int final = ((int)(unif_rand() * (upper - lower + 1)) + lower);
 PutRNGstate();
   return final;
}




//Constructor for ANNGA.default
ANNTraining::ANNTraining(int nbLayers,int * neuronPerLayer,int lengthData,double **tmatIn,double **tmatOut,int iMaxPopulation,double dmutRate,double dcrossRate,double dminW,double dmaxW, int iMaxGenerationSameResult,bool bMaxGenerationSameResult,double passSigma, double passProbGauss, bool rprintBestChromosome){		
			mPopulationSize		= iMaxPopulation;
			mfMutationRate		= dmutRate;
			maxGenerationSameResult	= iMaxGenerationSameResult;
			boolMaxGenerationSameResult	= bMaxGenerationSameResult;
			mfCrossoverRate		= crossRate;
			mLayerNum		= nbLayers;
			mGenerationNumber	= 0;
			trainInput		= 0;
			desiredOutput		= 0;
			MaxPopulation		=iMaxPopulation;
			mutRate			=dmutRate;
			crossRate		=dcrossRate;	
			minW			=dminW;
			maxW			=dmaxW;	
			meanFitness		=0;
			sigma			=passSigma;
			probGauss		=passProbGauss;
			printBestChromosome =rprintBestChromosome;
			rateModificationMutRate=0.9;
			lastGenerationBest=999999999; //initialisation to this number because no population should be that size
			generationSameResult=0;
			



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
			 mFitnessValues = new double[mPopulationSize]; //fitness i is the fitness value related to i th individual mChromosomes[i] 

			 //create population vectors
			 mChromosomes = new Chromosome[mPopulationSize];
			int ii;
			#pragma omp parallel for private(ii) 
			 for(ii = 0 ; ii < mPopulationSize ; ii++){
				 mChromosomes[ii] = new double[mWeightConNum];
				mFitnessValues[ii] = 0; //CAREFUL!! maybe u should initialize to some other value
			 }
			 
			 //read training data to memory


			nbOfData		= lengthData;
			nbOfInput		= neuronPerLayer[0];
			nbOfOutput		= neuronPerLayer[nbLayers-1];


			dataIn = new double*[nbOfData];
			dataOut = new double*[nbOfData];
			outputANN = new double*[nbOfData];

			#pragma omp parallel for private(ii)
			for(ii = 0 ; ii < nbOfData ; ii++){
				dataIn[ii] = new double[nbOfInput];
				dataOut[ii] = new double[nbOfOutput];
				outputANN[ii] = new double[nbOfOutput];
				for(int j = 0 ; j < nbOfInput ; j++){
					dataIn[ii][j]  = tmatIn[ii][j];
				}
				for(int j = 0 ; j < nbOfOutput ; j++){
					dataOut[ii][j] = tmatOut[ii][j];
				}
	
			}

			vectorFitness.clear();



			
}

//Constructor for predict.ANN
ANNTraining::ANNTraining(int nbLayers,int * neuronPerLayer,int lengthData,double **tmatIn){
		
			mLayerNum		= nbLayers;
			rateModificationMutRate=0.9;
			//create neural network 
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


			nbOfData		= lengthData;
			nbOfInput		= neuronPerLayer[0];
			nbOfOutput		= neuronPerLayer[nbLayers-1];

			//allocate memory for input data - represented as 2D matrix - 
			dataIn = new double*[nbOfData];
			for(int i = 0 ; i < nbOfData ; i++)
				dataIn[i] = new double[nbOfInput];

			//allocate memory for output data - represented as 2D matrix - 

			outputANN = new double*[nbOfData];
			for(int i = 0 ; i < nbOfData ; i++){
				outputANN[i] = new double[nbOfOutput];
			}
			for(int i = 0 ; i < nbOfData ; i++){
				for(int j = 0 ; j < nbOfInput ; j++){
					dataIn[i][j]  = tmatIn[i][j];
				}
			}
	
}




ANNTraining::~ANNTraining(){}


////////////////////////////////////////////
void ANNTraining::calculateAllFitnessOfPopulation (){
	
	meanFitness=0;
	for(int i = 0 ; i < mPopulationSize ; i++){
		mFitnessValues[i] =getFitness (mChromosomes[i]);
		meanFitness= meanFitness + mFitnessValues[i];
	}
	meanFitness=meanFitness/(double)mPopulationSize;

	//find the best individual and save it
	double best = 999999999;		
	double worst = 0;
	int tempB = 0;
	int tempW = 0;


	for(int i = 0 ; i < mPopulationSize ; i++){ 
		if(mFitnessValues[i] < best){
			best = mFitnessValues[i];
			tempB = i;
		}
		if(mFitnessValues[i] > worst){
			worst = mFitnessValues[i];
			tempW = i;
		}
	}
	
	bestIndividual  = tempB;
	worstIndividual  = tempW;
	worstIndividual  = 2; 
	bestChromosome=mChromosomes[bestIndividual];	                             
}
////////////////////////////////////////////

void ANNTraining::printFitness (){
	cout<<"generation: "<<mGenerationNumber<<"  min fitness value: "<<getMinFitness () <<endl;
}


////////////////////////////////////////////
void  ANNTraining::initializePopulation(){

	
	int i;
	 
	for(i = 0 ; i < mPopulationSize ; i++){
		for(int j = 0 ; j < mWeightConNum ; j++){
			mChromosomes[i][j] = unifRand(minW, maxW);
		}
	}

	calculateAllFitnessOfPopulation ();
	


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
	#pragma omp  parallel for   private(i) 
	for( i = 0 ; i < mWeightConNum ; i++){
		diff[i] = mChromosomes[r1][i] + (mutRate * (mChromosomes[r3][i] - mChromosomes[r2][i]));
	}
}
////////////////////////////////////////////
void	ANNTraining::crossover(int v){

	double ran;
	int i;
	#pragma omp  parallel for   private(i,ran) 
	for( i = 0 ; i < mWeightConNum ; i++){
 		GetRNGstate();
		ran = unif_rand();
		PutRNGstate();
		if(ran < crossRate){
			crossedTrialIndividual[i] = diff[i];
		}else{
			crossedTrialIndividual[i] = mChromosomes[v][i];
		}
	}
}
////////////////////////////////////////////
void		ANNTraining::select(int v){
	if(getFitness (crossedTrialIndividual) < getFitness (mChromosomes[v])){
		int i;
		#pragma omp  parallel for   private(i) 
		for( i = 0 ; i < mWeightConNum ; i++){
			mChromosomes[v][i] =crossedTrialIndividual[i];
		}	
	}	
}


void	ANNTraining::crossoverGauss(int v){
	//if(!(bestIndividual==v)){
	double ran;
	int i;
	#pragma omp  parallel for   private(i,ran) 
	for(i = 0 ; i < mWeightConNum ; i++){
 		GetRNGstate();
		ran = unif_rand();
		PutRNGstate();
		if(ran < probGauss){
			GetRNGstate();
			crossedTrialIndividual[i]=mChromosomes[v][i]+norm_rand()*sigma;
			PutRNGstate();
		}else{
			crossedTrialIndividual[i]=mChromosomes[v][i];
		}
	}
	//}
}
void	ANNTraining::crossoverGaussBest(int v){
	//if(!(bestIndividual==v)){
	double ran;
	int i;
	#pragma omp  parallel for   private(i,ran) 
	for(i = 0 ; i < mWeightConNum ; i++){
		GetRNGstate();
		ran = unif_rand();
		PutRNGstate();
		if(ran < probGauss){
			GetRNGstate();
			crossedTrialIndividual[i]=mChromosomes[bestIndividual][i]+norm_rand()*sigma;
			PutRNGstate();
		}else{
			crossedTrialIndividual[i]=mChromosomes[bestIndividual][i];
		}
	}
	//}
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
void		ANNTraining::setANNweightsWithBestChromosome(){/////////////////////////////////////////////should 
ann->loadWights (bestChromosome);	
}
////////////////////////////////////////////
void ANNTraining::getANNresult(){

	ann->loadWights (mChromosomes[bestIndividual]);
	for(int i = 0 ; i < nbOfData ; i++){  
		ann->feedForward (dataIn[i]);
		ann->getOutput(i);
		for(int j = 0 ; j < nbOfOutput ; j++){
			outputANN[i][j] = ann->getOutput(j);
		}
	}
}
////////////////////////////////////////////
void ANNTraining::predictANN(){

	for(int i = 0 ; i < nbOfData ; i++){       
		ann->feedForward (dataIn[i]);
		ann->getOutput(i);
		for(int j = 0 ; j < nbOfOutput ; j++){
			outputANN[i][j] = ann->getOutput(j);
		}
	}
}

//each cycle consist on mutation,crossover and select operations for all individual
void		ANNTraining::cycle (bool print = false){
Rprintf ("\n***cycle***\n");
	for(int i = 0 ; i < mPopulationSize ; i++){
			mutate(i);
			crossover(i);
			select(i);
	}

	calculateAllFitnessOfPopulation();
	
	if(print==true){
		printFitness (bestIndividual);
	}

	if(lastGenerationBest==mFitnessValues[bestIndividual]){
		generationSameResult+=1;
	}else{
		generationSameResult=1;
		lastGenerationBest=mFitnessValues[bestIndividual];
	}

	if(boolMaxGenerationSameResult==true){
		if(generationSameResult==maxGenerationSameResult){
			Rprintf("Old mutation Rate %4.8f  \n", mfMutationRate);
			mfMutationRate=mfMutationRate*rateModificationMutRate;
			Rprintf("New mutation Rate %4.8f  \n", mfMutationRate);
		}
	}
	++mGenerationNumber;
	vectorFitness.push_back (mFitnessValues[bestIndividual]);
}

void		ANNTraining::cycleGauss (bool print = false){

Rprintf ("\n***cycleGauss***\n");

	for(int i = 0 ; i < mPopulationSize ; i++){
			crossoverGauss(i);
			select(i);
	}

	calculateAllFitnessOfPopulation();

	for(int i = 0 ; i < mPopulationSize ; i++){
			mutate(i);
			crossover(i);
			select(i);
	}

	calculateAllFitnessOfPopulation();


	if(print==true){
		printFitness (bestIndividual);
	}

	if(lastGenerationBest==mFitnessValues[bestIndividual]){
		generationSameResult+=1;
	}else{
		generationSameResult=1;
		lastGenerationBest=mFitnessValues[bestIndividual];
	}

	if(boolMaxGenerationSameResult==true){
		if(generationSameResult>=maxGenerationSameResult){
			Rprintf("\n\n\n\n*********************\nOld mutation Rate %4.8f  \n", mfMutationRate);
			mfMutationRate=mfMutationRate*rateModificationMutRate;
			Rprintf("New mutation Rate %4.8f  \n*********************\n\n\n\n\n", mfMutationRate);
			generationSameResult=0;
		}
	}
	++mGenerationNumber;
	vectorFitness.push_back (mFitnessValues[bestIndividual]);
}

void		ANNTraining::cycleGaussBest (bool print = false){
Rprintf ("\n***cycleGaussBest***\n");

	for(int i = 0 ; i < mPopulationSize ; i++){
		if( !(i == bestIndividual)){
			crossoverGaussBest(i);
		}
	}

	calculateAllFitnessOfPopulation();

	for(int i = 0 ; i < mPopulationSize ; i++){
			mutate(i);
			crossover(i);
			select(i);
	}

	calculateAllFitnessOfPopulation();

	if(print==true){
		printFitness (bestIndividual);
	}

	if(lastGenerationBest==mFitnessValues[bestIndividual]){
		generationSameResult+=1;
	}else{
		generationSameResult=1;
		lastGenerationBest=mFitnessValues[bestIndividual];
	}

	if(boolMaxGenerationSameResult==true){
		if(generationSameResult>=maxGenerationSameResult){
			Rprintf("\n\n\n\n*********************\nOld mutation Rate %4.8f  \n", mfMutationRate);
			mfMutationRate=mfMutationRate*rateModificationMutRate;

			Rprintf("New mutation Rate %4.8f  \n*********************\n\n\n\n\n", mfMutationRate);
			generationSameResult=0;
		}
	}
	++mGenerationNumber;
	vectorFitness.push_back (mFitnessValues[bestIndividual]);
}


void ANNTraining::printFitness (int i){

	printf("Generation:  %d  Best population fitness : %4.8f  Mean of population:%4.8f  ", mGenerationNumber, mFitnessValues[i],meanFitness);

	if(printBestChromosome==true){	
		printf("\n Best chromosome->");
		for(int j = 0 ; j < mWeightConNum ; j++){
				printf("%4.2f/", mChromosomes[i][j]);
		}
	}
	printf("\n");
}


////////////////////////////////////////////
double ANNTraining::getMinFitness(){	

	double min = mFitnessValues[0];
	double max = mFitnessValues[0];//mine

	for(int i = 1 ; i < mPopulationSize ; i++){
		if(mFitnessValues[i] < min){
			min = mFitnessValues[i];
			bestIndividual = i;
		}

		if(mFitnessValues[i] > max){
			max = mFitnessValues[i];
			worstIndividual = i;
		}
	}

	bestChromosome = mChromosomes[bestIndividual];
	return min;
}

////////////////////////////////////////////
void		ANNTraining::release (){
	delete [] mChromosomes;
	delete [] mFitnessValues;
	delete [] mNeuronNum; 
	ann->release ();
}
//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////





#endif
