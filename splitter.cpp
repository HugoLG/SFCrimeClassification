#include <fstream>
#include <iostream>
#include <algorithm>
#include <time.h>
#include <stdlib.h>
#include <stdio.h>

using namespace std;
/*
 * This program splits the given data
 * into training and validation data.
 */
int main(){
	srand(time(NULL));
	ifstream data;
	ofstream training;
	ofstream validation;
	data.open("firstColumnRemoved.csv");
	training.open("training.txt");
	validation.open("validation.txt");
	double a;
	int random;
	int i=0;
	int tr, val, tes;
	tr=val=tes=0;
	string s;
	while (getline(data, s)){
		i++;
		random = rand()%20;
		//randomly picks data points for
		//training or validation
		if(random<16){
			training<<s<<endl;
			tr++;
		}
		else{
			validation<<s<<endl;
			val++;
		}
	}
	data.close();
	training.close();
	validation.close();
	return 0;
}
