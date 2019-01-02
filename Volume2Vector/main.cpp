#pragma once
#include <iostream>
#include <cstdlib>
#include <string>
#include <vector>
#include <fstream>
#include <ctime>
#include <omp.h>
#include <vector>
#include <algorithm>
#include "Volume.h"
#include "Volume2Vector.h"
#include "MultiVolume.h"
using namespace std;

#define MAX_STRING 100
#define EXP_TABLE_SIZE 1000
#define MAX_EXP 6
#define MAX_SENTENCE_LENGTH 100000000
#define MAX_CODE_LENGTH 100
#define MAX_DIMENSION 256
const int vocab_hash_size = 2560000;  // Maximum 30 * 0.7 = 21M words in the vocabulary

typedef double real;                    // Precision of float numbers
										//typedef unsigned short ORIGIN_DATA_TYPE;
typedef unsigned char ORIGIN_DATA_TYPE;

int arg_pos(char *str, int argc, char **argv) {
	for (auto a = 1; a < argc; a++)
	{
		if (!strcmp(str, argv[a])) 
		{
			if (a == argc - 1) 
			{
				printf("Argument missing for %s\n", str);
				exit(1);
			}
			return a;
		}
		
	}
	return 0xffffffff;
}

void readInfoFile(string& infoFileName, int& data_number, string& datatype, int3& dimension, vector<string>& filelist) {
	
	ifstream inforFile(infoFileName);


	inforFile >> data_number;
	inforFile >> datatype;
	inforFile >> dimension.x >> dimension.y >> dimension.z;

	const string filePath = infoFileName.substr(0, infoFileName.find_last_of('/')+1);
	cout << filePath << endl;

	for (int i = 0; i < data_number; i++)
	{
		string rawFileName;
		inforFile >> rawFileName;
		string volumePath = filePath + rawFileName;
		filelist.push_back(volumePath);
	}
}


int main(int argc, char** argv)
{

	int vocab_size = 0;
	int layer1_size = 100;				//size of word vector
	int hs = 1;						//Use Hierarchical Softmax; default is 0 (not used)
	int negative = 0;				//Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
	long long train_words = 0;
	int debug_mode = 2;				//Set the debug mode (default = 2 = more info during training)
	int min_count = 0;				//This will discard words that appear less than <int> times; default is 5
	int histogramDimension = 128;
	int classes = 3;
	int distance2 = 1;
	int random_iteration = 27 - 1;	//Number of random iterations; default is 26, cube of 3*3*3-1. 
	int isVoxelBasedHistogramSaved = 1;
	int output_mode = 3;				//Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
	real sample = -1e-3;			//Set threshold for occurrence of words. Those that appear with higher frequency in the training data
	real alpha = 0.0025;				//Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	int sample_mode = 0;			//The mode of sample. 0 is voxel-based and 1 for the value-based.
	bool onlyOutOfRangeNegativeUsed = true;
	char train_file[100];

	int i;
	if (argc == 1) {
		printf("VOLUME 2 VECTOR estimation toolkit\n\n");
		printf("Options:\n");
		printf("Parameters for training:\n");
		printf("\t-train <file>\n");
		printf("\t\tUse text data from <file> to train the model\n");
		printf("\t-output <file>\n");
		printf("\t\tUse <file> to save the resulting word vectors / word clusters\n");
		printf("\t-size <int>\n");
		printf("\t\tSet size of word vectors; default is 100\n");
		printf("\t-sample <float>\n");
		printf("\t\tSet threshold for occurrence of words. Those that appear with higher frequency in the training data\n");
		printf("\t\twill be randomly down-sampled; default is 1e-3, useful range is (0, 1e-5)\n");
		printf("\t-hs <int>\n");
		printf("\t\tUse Hierarchical Softmax; default is 0 (not used)\n");
		printf("\t-negative <int>\n");
		printf("\t\tNumber of negative examples; default is 5, common values are 3 - 10 (0 = not used)\n");
		printf("\t-iter <int>\n");
		printf("\t\tRun more training iterations (default 5)\n");
		printf("\t-min_count <int>\n");
		printf("\t\tThis will discard words that appear less than <int> times; default is 0\n");
		printf("\t-alpha <float>\n");
		printf("\t\tSet the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW\n");
		printf("\t-classes <int>\n");
		printf("\t\tThe number of k-means cluster. default =3\n");

		printf("\t-save-vocab <file>\n");
		printf("\t\tThe vocabulary will be saved to <file>\n");
		printf("\t-read-vocab <file>\n");
		printf("\t\tThe vocabulary will be read from <file>, not constructed from the training data\n");
		printf("\t-distance <int>\n");
		printf("\t\tThe window of the volume. distance=1(surrounding 6 voxel of the center voxel);distance=2(surrounding 18 voxel of the center voxel);distance=3(surrounding 26 voxel of the center voxel)\n");

		printf("\t-sample_mode <int>\n");
		printf("\t\tThe mode of sample. 0 is voxel-based and 1 for the value-based.\n");
		printf("\t-random_sample_number <int>\n");
		printf("\t\tNumber of random iterations; default is 26, cube of 3*3*3-1.\n");
		printf("\t-onlyOutOfRangeNegativeUsed <bool>\n");
		printf("\t\tWhether use the out of range negative value\n");
		printf("\t-vector_size <int>\n");
		printf("\t\tThe dimension of vector size\n");
		
		printf("\t-histogram_size <int>\n");
		printf("\t\tThe histogram size\n");
		printf("\t-voxelBasedHistogramSaved <int>\n");
		printf("\t\tWhether the histogram based voxels saved. default = 0 no save; = 1 save. The file will be saved as .csv\n");
		printf("\nExamples:\n");
		printf("./word2vec -train data.txt -output vec.txt -size 200 -window 5 -sample 1e-4 -negative 5 -hs 0 -iter 3\n\n");
		printf("./word2vec.exe - train 'J:/science data/4 Combustion/jet_0051/jet_vort_0051.dat' -output_mode 3 -distance 3 -sample_mode 1 -dimension 128 -negative 6\n\n");
		//return 0;

	}
	//if ((i = arg_pos(static_cast<char *>("-save-vocab"), argc, argv)) > 0) strcpy(save_vocab_file, argv[i + 1]);
	//if ((i = arg_pos(static_cast<char *>("-read-vocab"), argc, argv)) > 0) strcpy(read_vocab_file, argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-debug"), argc, argv)) != 0xffffffff) debug_mode = atoi(argv[i + 1]);
	
	if ((i = arg_pos(static_cast<char *>("-train"), argc, argv)) != 0xffffffff) strcpy(train_file, argv[i + 1]);
	else
	{
		strcpy(train_file, "F:/CThead/head.vifo");
	}
	
	if ((i = arg_pos(static_cast<char *>("-alpha"), argc, argv)) != 0xffffffff) alpha = atof(argv[i + 1]);
	//if ((i = arg_pos(static_cast<char *>("-output"), argc, argv)) > 0) strcpy(output_file, argv[i + 1]);
	//if ((i = arg_pos(static_cast<char *>("-window"), argc, argv)) > 0) window = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-sample"), argc, argv)) != 0xffffffff) sample = atof(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-hs"), argc, argv)) != 0xffffffff) hs = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-negative"), argc, argv)) != 0xffffffff) negative = atoi(argv[i + 1]);

	if ((i = arg_pos(static_cast<char *>("-min_count"), argc, argv)) != 0xffffffff) min_count = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-classes"), argc, argv)) != 0xffffffff) classes = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-distance"), argc, argv)) != 0xffffffff) distance2 = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-histogram_size"), argc, argv)) != 0xffffffff) histogramDimension = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-sample_mode"), argc, argv)) != 0xffffffff) sample_mode = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-random_sample_number"), argc, argv)) != 0xffffffff) random_iteration = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-vector_size"), argc, argv)) != 0xffffffff) layer1_size = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-voxelBasedHistogramSaved"), argc, argv)) != 0xffffffff) isVoxelBasedHistogramSaved = atoi(argv[i + 1]);
	if ((i = arg_pos(static_cast<char *>("-min_count"), argc, argv)) != 0xffffffff) min_count = atoi(argv[i + 1]);


	srand(time(nullptr));
	const auto time_before = clock();
	string infoFileName(train_file);
	int data_number;
	string datatype;
	int3 dimension;
	vector<string> filelist;
	readInfoFile(infoFileName, data_number, datatype, dimension, filelist);

	vector<vector<vector<int>>> regularData;
	cout << datatype << endl;
	if(datatype == "float")
	{
		auto* volume = new Volume<float>(histogramDimension);
		volume->setFileAndRes(filelist, dimension);
		regularData = volume->getRegularData();
	}
	else if(datatype == "ushort")
	{
		auto* volume2 = new Volume<unsigned short>(histogramDimension);
		volume2->setFileAndRes(filelist, dimension);
		regularData = volume2->getRegularData();
	}
	else if(datatype == "uchar")
	{
		auto* volume3 = new Volume<unsigned char>(histogramDimension);
		volume3->setFileAndRes(filelist, dimension);
		regularData = volume3->getRegularData();
	}
	else
	{
		cout << "Read data error!" << endl;
		return -1;
	}

	//auto volume2vector = new Volume2Vector(regularData, volume->getDimension(), volume->getHistogramDimension());
	auto volume2vector = new Volume2Vector();

	volume2vector->setInformation(regularData, dimension, histogramDimension,
		layer1_size, distance2, negative, random_iteration, isVoxelBasedHistogramSaved, min_count, output_mode,
		sample, alpha, sample_mode, output_mode, classes,hs);

	volume2vector->learnVocab();
	volume2vector->init();
	volume2vector->train();
	volume2vector->save();
	const auto time_end = clock();
	cout << "Total time : " << (time_end - time_before)*1.0f/1000.0f <<"s"<< endl;
	return 0;
}