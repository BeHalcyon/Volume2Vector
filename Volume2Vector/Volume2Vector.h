#pragma once
#include <vector>
#include "Neuron.h"
#include <iostream>
#include "Volume.h"
using namespace std;
typedef double real;

class Volume2Vector
{
public:
	Volume2Vector();

	void setInformation(vector<vector<vector<int>>>& regularData, int3& dimension, int histogramDimension,
	                    int layer1_size,
	                    int distance2, int negative, int random_iteration, int isVoxelBasedHistogramSaved,
	                    int min_count,
	                    int output_mode, real sample, real alpha, int sample_mode, bool onlyOutOfRangeNegativeUsed,
	                    int classes, int hs);


	void init();
	~Volume2Vector();
	void setRegularVolume(vector<vector<vector<int>>>& regularData, int3& dimension);
	int getWordHash(int intword) const;
	int addWordToVocab(int intword);
	int searchVocab(int intword);
	void learnVocab();
	void sortVocab();
	void initUnigramTable();

	void initNet();
	void createBinaryTree();
	void cbowGram(long long index, vector<int>& sen, int distance2, vector<vector<int>>& histogramBasedVoxel);

	void train();
	void save();
	real similarity(int index_a, int index_b);
	void saveSimilarityArray(string& filepath);

private:

	vector<real> syn0;
	vector<real> syn1;
	vector<real> syn1neg;
	vector<Neuron> vocab;
	vector<int> table;
	vector<vector<vector<int>>> regularData;
	int3 dimension;
	vector<int> vocab_hash;

	vector<real> expTable;


	int vocab_size = 0;
	int layer1_size = 100;				//size of word vector
	int hs = 1;						//Use Hierarchical Softmax; default is 1 (used)
	int negative = 0;				//Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
	long long train_words = 0, word_count_actual = 0;
	int debug_mode = 2;				//Set the debug mode (default = 2 = more info during training)
	int min_count = 0;				//This will discard words that appear less than <int> times; default is 5
	int histogramDimension = 256;
	int classes = 3;				//K-means cluster number: default =3;
	int distance2 = 1;
	int random_iteration = 27-1;	//Number of random iterations; default is 26, cube of 3*3*3-1. 
	int isVoxelBasedHistogramSaved = 1;
	int output_mode{};				//Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
	real sample = -1e-3;			//Set threshold for occurrence of words. Those that appear with higher frequency in the training data
	real alpha = 0.0025;				//Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
	int sample_mode = 0;			//The mode of sample. 0 is voxel-based and 1 for the value-based.
	bool onlyOutOfRangeNegativeUsed = true;
	real starting_alpha;
	int iter = 5;					//Run more training iterations (default 5)???为什么这么少
	const int table_size = 1e8;
	const int vocab_hash_size = 2560000;
	const int exp_table_size = 1000;
	const int max_exp = 6;

};

