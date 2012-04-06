////////////////////////////////////////////////////////////////////
// ANN.h: Artificial Neural Network optimized by Genetic Algorithm 
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
#ifndef ARTIFICIAL_NEURAL_NETWORK_H
#define ARTIFICIAL_NEURAL_NETWORK_H

class ArtificialNeuralNetwork{


public:
	ArtificialNeuralNetwork(int numOfLay, int* numOfNeurunsInEachLayer);
	~ArtificialNeuralNetwork();
	
	void					feedForward(double* input);	
	double					getOutput(int index);	//gets the output of i th neuron in output layer
	double  				getMeanSquareError(double* target);		//gets the MSE value of the net
	void					loadWights(double*);
	void					release();

	/***************************************************************************/
	/* Neuron values are stored in a 2D array.Simply, [2][3] means the value of*/
	/*  neuron which is 3th layer and 4th neuron. (indexing starts from 0)     */
	/***************************************************************************/
	double **				m_neuronValues; 

	/***************************************************************************/
	/* weights between neurons.It is represented with a 3D array.Indexing      */
	/* is [2][4][5] means weight between 4th neuron in second hidden layer and */
	/* 5th neuron in the previous (1st) hidden layer.(indexing starts from 0)  */
	/***************************************************************************/
	double ***				m_neuronWeights;

	int					m_numberOfLayers; //including input layer
	int*					m_numOfNeurons;   //number of neurons in each layer

	inline double 				sigmoid(double inValue);
 	inline double  				step(double);




};
#endif
