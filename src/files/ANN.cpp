////////////////////////////////////////////////////////////////////
// ANN.cpp: Artificial Neural Network optimized by Genetic Algorithm 
//
// Copyright (C) 2009, 2010 Emre Caglar
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
#ifndef ARTIFICIAL_NEURAL_NETWORK_CPP
#define ARTIFICIAL_NEURAL_NETWORK_CPP
#include "ANN.h"
#include <cstdlib>
#include <cmath>
#include <fstream>
#include <R.h>

using namespace std;

ArtificialNeuralNetwork::ArtificialNeuralNetwork(int numOfLay, int *numOfNeurunsInEachLayer){


	m_numberOfLayers	= numOfLay;

	//save neuron numbers in each layer
	m_numOfNeurons = new int[m_numberOfLayers];
	m_neuronValues = new double*[m_numberOfLayers];
	for(int i = 0 ; i < m_numberOfLayers ; i++){
		//save neuron numbers in each layer
		m_numOfNeurons[i] = numOfNeurunsInEachLayer[i];
		//allocate memory for neuron values - represented as 2D matrix -
		m_neuronValues[i] = new double[m_numOfNeurons[i]];
	}
	 

	//allocate memory for weights
	m_neuronWeights = new double **[m_numberOfLayers];
	for(int i = 1 ; i < m_numberOfLayers ; i++){  
		m_neuronWeights[i] = new double*[m_numOfNeurons[i]];
		for(int j = 0 ; j < m_numOfNeurons[i] ; j++){
			m_neuronWeights[i][j] = new double[m_numOfNeurons[i-1]+1];
			for(int k = 0 ; k < m_numOfNeurons[i-1]+1 ; k++){
				m_neuronWeights[i][j][k]=0;   
			}
		}
	}
}
////////////////////////////////////////////
ArtificialNeuralNetwork::~ArtificialNeuralNetwork(){}
////////////////////////////////////////////
void ArtificialNeuralNetwork::release (){

	for (int i = 1; i < m_numberOfLayers; ++i) {
		for (int j = 0; j < m_numOfNeurons[i]; ++j){
			delete [] m_neuronWeights[i][j];
		}
		delete [] m_neuronWeights[i];
	}
	delete [] m_neuronWeights;


	for(int i  = 0 ; i < m_numberOfLayers ; ++i)
		delete[] m_neuronValues[i];
	delete[] m_neuronValues;

	delete[] m_numOfNeurons;

}

////////////////////////////////////////////
void ArtificialNeuralNetwork::feedForward(double *input){

	double sum;
	//first set input layer with input pattern
	for(int i = 0 ; i < m_numOfNeurons[0] ; i++){
		m_neuronValues[0][i] = input[i];
	}

	//then feed forward it
	//for each layer in neural network
	for(int i = 1 ; i < m_numberOfLayers ; i++){

		//for each neuron on a particular layer
		for(int j = 0 ; j < m_numOfNeurons[i] ; j++){

			sum = 0.0; //reset sum
			for(int k = 0 ; k < m_numOfNeurons[i-1] ; k++){
				sum += m_neuronValues[i-1][k] * m_neuronWeights[i][j][k];
			}
			//add bias
			sum +=m_neuronWeights[i][j][m_numOfNeurons[i-1]];

			m_neuronValues[i][j] = sigmoid(sum);
			
		}
	}

}




////////////////////////////////////////////
double ArtificialNeuralNetwork::getOutput(int index){
	
	return m_neuronValues[m_numberOfLayers-1][index];
}
////////////////////////////////////////////
double ArtificialNeuralNetwork::getMeanSquareError(double* target){

		double mse=0;
		int i;
		//#pragma omp  parallel for   private(i) reduction(+: mse) 
		//add all the output
		for( i = 0 ; i < m_numOfNeurons[m_numberOfLayers-1] ; i++){
			mse = mse + (target[i]-m_neuronValues[m_numberOfLayers-1][i])*(target[i]-m_neuronValues[m_numberOfLayers-1][i]);
		}
		
		return mse;
	
}
////////////////////////////////////////////
double ArtificialNeuralNetwork::sigmoid(double inValue){
	
	return 1.0/(1+exp(-inValue));

}
////////////////////////////////////////////
double ArtificialNeuralNetwork::step(double inValue){
	
	if(inValue>=0){return 1;}else{return 0;}
}
////////////////////////////////////////////
void ArtificialNeuralNetwork::loadWights(double *weightVector){

	for(int layer = 1; layer < m_numberOfLayers ; ++layer){ 
		for(int neuron = 0 ; neuron < m_numOfNeurons[layer] ; neuron++){
			for(int preNeuron = 0 ; preNeuron < m_numOfNeurons[layer-1] ; preNeuron++){
				m_neuronWeights[layer][neuron][preNeuron] = *(weightVector++); 
			}
			//load bias
			m_neuronWeights[layer][neuron][m_numOfNeurons[layer-1]] = *(weightVector++);
		}
	}
}
////////////////////////////////////////////

#endif

