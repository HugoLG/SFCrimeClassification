#include <iostream>
#include "neuralNetwork.h"

int main(int argc, char *argv[]) {
	//Command line arguments are used in order to facilitate
	//the use of the program from the python interface
	if (argc != 2 || (argv[1][0]!='A' && argv[1][0]!='B' && argv[1][0]!='C')) {
		cout << "Invalid usage.\n" << "Correct usage: \n" << "./network option\n";
		cout << "Valid options: \n" << "A: Train the classifier\n" << "B: Run the classifier over the test file\n" << "C: Predict a crime\n";
		cout << "The first time the program is run, the A parameter must be called first.\n";
		cout << "The program will not run over the test file or predict a crime if it has not been trained first.\n";
		exit(0);
	}
	//The parameter A will train the network
	if (argv[1][0] == 'A') {
		//Dynamic memory is used here to speed up 
		//the training process. Since the data set 
		//is too big, the program would'nt allow
		//for a normal memory allocation
		double** data = new double*[900000];
		for (int i = 0; i < 900000; i++) {
			data[i] = new double[77];
		}
		double** validationData = new double*[200000];
		for (int i = 0; i < 200000; i++) {
			validationData[i] = new double[77];
		}

		//Constructor parameters: lambda, eta, momentum, input neurons,
		//hidden layers, hidden neurons per layer, output neurons
		NeuralNetwork network(0.6, 0.8, 0.0, 38, 1, 25, 39);
		//The data and validation dynamic arrays
		//are passed to the network. The network
		//uses them by reference
		network.run(150, data, validationData);
		network.saveToFile("saved.txt");

		//Free memory
		for (int i = 0; i < 900000; i++) {
			delete[] data[i];
		}
		delete[] data;
		for (int i = 0; i < 200000; i++) {
			delete[] validationData[i];
		}
		delete[] validationData;
	}
	else if (argv[1][0] == 'B') {
		double** data = new double*[900000];
		for (int i = 0; i < 900000; i++) {
			data[i] = new double[77];
		}
		NeuralNetwork network("saved.txt");
		network.runTest(data);
		for (int i = 0; i < 900000; i++) {
			delete[] data[i];
		}
		delete[] data;
	}
	else if (argv[1][0] == 'C') {
		NeuralNetwork network("saved.txt");
		network.predict();
	}
	cout << "Finished\n";
	return 0;
}
