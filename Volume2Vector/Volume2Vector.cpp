#include "Volume2Vector.h"
#include "Volume.h"
#include <algorithm>
#include <ctime>


Volume2Vector::Volume2Vector(): starting_alpha(0)
{
	vocab_hash.resize(vocab_hash_size);

	expTable.reserve(exp_table_size + 1);
	expTable.resize(exp_table_size + 1);
	for (auto i = 0; i < exp_table_size; i++)
	{
		expTable[i] = exp((i / static_cast<real>(exp_table_size) * 2 - 1) * max_exp); // Precompute the exp() table
		expTable[i] = expTable[i] / (expTable[i] + 1); // Precompute f(x) = x / (x + 1)
	}
}

/**
 * \brief 
 * \param regularData : origin data with range from between 0 to histogramDimension
 * \param dimension : the dimension of origin data
 * \param histogramDimension : the histogram dimension
 * \param layer1_size : the vector length of one word in the vocab
 * \param distance2 : the window of the volume. distance=1(surrounding 6 voxel of the center voxel);distance=2(surrounding 18 voxel of the center voxel);distance=3(surrounding 26 voxel of the center voxel)
 * \param negative : Number of negative examples; default is 0, common values are 3 - 10 (0 = not used)????
 * \param random_iteration : Number of random iterations; default is 26, cube of 3*3*3-1. 
 * \param isVoxelBasedHistogramSaved 
 * \param min_count : This will discard words that appear less than <int> times; default is 5
 * \param output_mode : Output word classes rather than word vectors; default number of classes is 0 (vectors are written), =2 is the cluster, =3 is similarity array
 * \param sample : Set threshold for occurrence of words. Those that appear with higher frequency in the training data
 * \param alpha : Set the starting learning rate; default is 0.025 for skip-gram and 0.05 for CBOW
 * \param sample_mode : The mode of sample. 0 is voxel-based and 1 for the value-based.
 * \param onlyOutOfRangeNegativeUsed
 * \param classes : The number of k-means cluster. default =3.
 */
void Volume2Vector::setInformation(vector<vector<vector<int>>>& regularData, 
			int3& dimension, int histogramDimension,
			int layer1_size, int distance2, int negative, 
			int random_iteration, int isVoxelBasedHistogramSaved, int min_count,
			int output_mode, real sample, real alpha, int sample_mode,
			bool onlyOutOfRangeNegativeUsed, int classes, int hs)
{
	this->regularData = regularData;
	this->dimension = dimension;
	this->histogramDimension = histogramDimension;
	this->layer1_size = layer1_size;
	this->distance2 = distance2; 
	this->negative = negative;
	this->random_iteration = random_iteration;
	this->isVoxelBasedHistogramSaved = isVoxelBasedHistogramSaved;
	this->min_count = min_count;
	this->output_mode = output_mode;
	this->sample = sample;
	this->alpha = alpha;
	this->sample_mode = sample_mode;
	this->onlyOutOfRangeNegativeUsed = onlyOutOfRangeNegativeUsed;
	this->classes = classes;

	vocab.resize(histogramDimension);


	cout << "alpha:\t" << this->alpha << endl;
	cout << "sample:\t" << this->sample << endl;
	cout << "hs:\t" << this->hs << endl;
	cout << "negative:\t" << this->negative << endl;
	cout << "min_count:\t" << this->min_count << endl;
	cout << "classes:\t" << this->classes << endl;
	cout << "distance:\t" << this->distance2 << endl;
	cout << "histogram_size:\t" << this->histogramDimension << endl;
	cout << "sample_mode:\t" << this->sample_mode << endl;
	cout << "random_sample_number:\t" << this->random_iteration << endl;
	cout << "vector_size:\t" << this->layer1_size << endl;
	cout << "voxelBasedHistogramSaved:\t" << this->isVoxelBasedHistogramSaved << endl;
	cout << "min_count:\t" << this->min_count << endl;
}
void Volume2Vector::init()
{
	initNet();
	if(negative>0)
	{
		initUnigramTable();
	}
	for (auto i = 0;i < vocab_size;i++) {
		cout << vocab[i].intword << "\t" << vocab[i].cn << "\t" << vocab_hash[vocab[i].intword] << "\t" << vocab_hash[i] << "\t";
		for (auto j = 0;j < vocab[i].codelen;j++)
			cout << static_cast<int>(vocab[i].code[j]);
		cout << endl;
	}
}
Volume2Vector::~Volume2Vector()
{
	syn0.clear();
	syn1.clear();
	syn1neg.clear();
	vocab.clear();
	table.clear();
	expTable.clear();
	//regularData.clear();
	vocab_hash.clear();
}
void Volume2Vector::setRegularVolume(vector<vector<vector<int>>>& regularData, int3& dimension)
{
	this->regularData = regularData;
	this->dimension = dimension;
}
int Volume2Vector::getWordHash(const int intword) const
{
	return intword;
}

int Volume2Vector::addWordToVocab(int intword)
{
	vocab[vocab_size].intword = intword;
	vocab[vocab_size].cn = 0;
	vocab_size++;

	const unsigned int hash = getWordHash(intword);

	vocab_hash[hash] = vocab_size - 1;

	return vocab_size - 1;
}
/**
 * \brief Returns position of a word in the vocabulary; if the word is not found, returns -1
 * \param intword 
 * \return 
 */
int Volume2Vector::searchVocab(const int intword) {
	const unsigned int hash = getWordHash(intword);
	return vocab_hash[hash];
}

void Volume2Vector::learnVocab()
{
	if(regularData.empty())
	{
		return;
	}

	int a;
	for (a = 0; a < vocab_hash_size; a++)
		vocab_hash[a] = -1;

	vocab_size = 0;

	//cout << "Debug information 20181120: " << dimension.x << " " << dimension.y << " " << dimension.z << endl;
	//readRawFile(train_file, data);

	addWordToVocab(0);
	for (auto k = 0; k < dimension.z; k++)
	{
		for (auto j = 0; j < dimension.y; j++)
		{
			for (auto i = 0; i < dimension.x; i++)
			{
				auto& intword = regularData[k][j][i];
				train_words++;
				if ((debug_mode > 1) && (train_words % 10000 == 0)) {
					printf("Loading data to vocab: %f%%%c", train_words*1.0f / (dimension.x*dimension.y*dimension.z) * 100, 13);
					fflush(stdout);
				}
				const auto index = searchVocab(intword);
				if (index <0) {
					a = addWordToVocab(intword);
					vocab[intword].cn = 1;
				}
				else vocab[index].cn++;
				//这里将大小变小，暂时删掉
				//if (vocab_size > vocab_hash_size * 0.7) ReduceVocab();
			}
		}
	}

	sortVocab();
	if (debug_mode > 0) {
		printf("Vocab size: %lld\n", static_cast<long long>(vocab_size));
		printf("Words in train file: %lld\n", train_words);
	}
	//cout << "Test end" << endl;
}

/**
 * \brief Sorts the vocabulary by frequency using word counts
 */
void Volume2Vector::sortVocab() {
	//if (!isIntValueUsed) return;
	int a;
	//sort(vocab.begin(), vocab.begin() + vocab_size, VocabCompare);
	sort(vocab.begin(), vocab.begin() + vocab_size);

	for (a = 0; a < vocab_hash_size; a++) vocab_hash[a] = -1;

	const int size = vocab_size;
	train_words = 0;
	//将重新匹配vocab_hash，以值出现的频率排序，频率越高，出现的顺序越前，即vocab_hash值越前
	for (a = 0; a < size; a++) {
		// Words occuring less than min_count times will be discarded from the vocab
		if ((vocab[a].cn < min_count) /*&& (a != 0)*/) {
			vocab_size--;
		}
		else {
			// Hash will be re-computed, as after the sorting it is not actual
			const unsigned int hash = getWordHash(vocab[a].intword);
			//记录该hash值在索引的位置。
			vocab_hash[hash] = a;
			train_words += vocab[a].cn;
		}
	}

}
void Volume2Vector::initUnigramTable()
{
	int a;
	double train_words_pow = 0;
	const auto power = 0.75;
	//table = (int *)malloc(table_size * sizeof(int));
	table.reserve(table_size);
	table.resize(table_size);
	for (a = 0; a < vocab_size; a++) train_words_pow += pow(vocab[a].cn, power);
	auto i = 0;
	auto d1 = pow(vocab[i].cn, power) / train_words_pow;
	for (a = 0; a < table_size; a++) {
		table[a] = i;
		if (a / static_cast<double>(table_size) > d1) {
			i++;
			d1 += pow(vocab[i].cn, power) / train_words_pow;
		}
		if (i >= vocab_size) i = vocab_size - 1;
	}
}


void Volume2Vector::initNet()
{
	long long a, b;
	unsigned long long next_random = 1;

	syn0.reserve(vocab_size * layer1_size);
	syn0.resize(vocab_size * layer1_size);

	if (hs) {
		syn1.reserve(vocab_size * layer1_size);
		syn1.resize(vocab_size * layer1_size);
		for (a = 0; a < vocab_size; a++) for (b = 0; b < layer1_size; b++)
			syn1[a * layer1_size + b] = 0;
	}

	if (negative>0) {
		syn1neg.reserve(vocab_size * layer1_size);
		syn1neg.resize(vocab_size * layer1_size);

		for (a = 0; a < vocab_size; a++)
			for (b = 0; b < layer1_size; b++)
				syn1neg[a * layer1_size + b] = 0;
	}
	for (a = 0; a < vocab_size; a++)
		for (b = 0; b < layer1_size; b++) {
			next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
			syn0[a * layer1_size + b] = (((next_random & 0xFFFF) / static_cast<real>(65536)) - 0.5) / layer1_size;
		}


	createBinaryTree();

}
void Volume2Vector::createBinaryTree() {
	long long a, b, i, min1i, min2i, pos1, pos2, point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	auto* count = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	auto* binary = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	auto* parent_node = static_cast<long long *>(calloc(vocab_size * 2 + 1, sizeof(long long)));
	for (a = 0; a < vocab_size; a++) count[a] = vocab[a].cn;
	for (a = vocab_size; a < vocab_size * 2; a++) count[a] = 1e15;
	pos1 = vocab_size - 1;
	pos2 = vocab_size;
	// Following algorithm constructs the Huffman tree by adding one node at a time
	for (a = 0; a < vocab_size - 1; a++) {
		// First, find two smallest nodes 'min1, min2'
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min1i = pos1;
				pos1--;
			}
			else {
				min1i = pos2;
				pos2++;
			}
		}
		else {
			min1i = pos2;
			pos2++;
		}
		if (pos1 >= 0) {
			if (count[pos1] < count[pos2]) {
				min2i = pos1;
				pos1--;
			}
			else {
				min2i = pos2;
				pos2++;
			}
		}
		else {
			min2i = pos2;
			pos2++;
		}
		count[vocab_size + a] = count[min1i] + count[min2i];
		parent_node[min1i] = vocab_size + a;
		parent_node[min2i] = vocab_size + a;
		binary[min2i] = 1;
	}
	// Now assign binary code to each vocabulary word
	for (a = 0; a < vocab_size; a++) {
		b = a;
		i = 0;
		while (1) {
			code[i] = binary[b];
			point[i] = b;
			i++;
			b = parent_node[b];
			if (b == vocab_size * 2 - 2) break;
		}
		vocab[a].codelen = i;
		vocab[a].point[0] = vocab_size - 2;
		for (b = 0; b < i; b++) {
			vocab[a].code[i - b - 1] = code[b];
			vocab[a].point[i - b] = point[b] - vocab_size;
		}
	}
	free(count);
	free(binary);
	free(parent_node);
}

void Volume2Vector::cbowGram(const long long index, vector<int>& sen,
	const int distance2, vector<vector<int>>& histogramBasedVoxel)
{
	long long word = 0;
	unsigned long long next_random = 1;
	int c, d;
	if (index >= sen.size()) {
		cout << "Error: array index out of bounds.\n" << endl;
		return;
	}
	word = sen[index];
	if (word == -1 || word >= histogramDimension) return;
	auto cw = 0;
	auto currentIndex = 0;
	const auto offset = static_cast<int>(sqrt(distance2));
	const auto xDim = dimension.x;
	const auto yDim = dimension.y;
	const auto zDim = dimension.z;
	//real *neu1 = (real *)calloc(layer1_size, sizeof(real));
	vector<real> neu1;
	neu1.reserve(layer1_size);
	neu1.resize(layer1_size);
	//real *neu1e = (real *)calloc(layer1_size, sizeof(real));
	vector<real> neu1e;
	neu1e.reserve(layer1_size);
	neu1e.resize(layer1_size);
	for (c = 0; c < layer1_size; c++) neu1[c] = 0;
	for (c = 0; c < layer1_size; c++) neu1e[c] = 0;

	const int k = index / (xDim*yDim);
	const int i = index%xDim;
	const int j = (index % (xDim*yDim)) / xDim;

	//找到当前块的最大值和最小值
	auto maxValue = -1;
	auto minValue = histogramDimension;
	int centerValue = regularData[k][j][i];

	//ReadIntWordFromArray(centerValue, i, j, k, data);
	auto random_sample_value = 0;
	vector<int> tempValueArray;
	for (auto zvalue = k - offset;zvalue <= k + offset;zvalue++) {
		if (zvalue<0 || zvalue >= regularData.size()) continue;
		//for y
		for (auto yvalue = j - offset;yvalue <= j + offset;yvalue++) {
			if (yvalue<0 || yvalue >= regularData[0].size()) continue;
			//for x
			for (auto xvalue = i - offset;xvalue <= i + offset;xvalue++) {
				if (xvalue<0 || xvalue >= regularData[0][0].size()) continue;

				if (abs(zvalue - k) + abs(yvalue - j) + abs(xvalue - i)>distance2) continue;
				if (zvalue == k&&yvalue == j&&xvalue == i) continue;
				currentIndex = zvalue*xDim*yDim + yvalue*xDim + xvalue;

				if (currentIndex>xDim*yDim*zDim - 1) continue;

				if (sen[currentIndex] == -1) continue;
				auto intword = vocab[sen[currentIndex]].intword;

				if (isVoxelBasedHistogramSaved)
				{
					histogramBasedVoxel[centerValue][intword]++;
				}

				if (maxValue < intword) maxValue = intword;
				//避免边界影响？
				if (minValue > intword && intword>0) minValue = intword;
				if (sample_mode == 0) {
					tempValueArray.push_back(intword);
					//cw++;
				}

			}
		}
	}
	if (sample_mode) {
		tempValueArray.clear();
		for (d = 0;d < random_iteration;d++) {
			auto randValue = rand() % (maxValue - minValue + 1) + minValue;
			if (vocab_hash[randValue] <0 || vocab_hash[randValue] >= histogramDimension) continue;
			for (c = 0; c < layer1_size; c++) {
				//Debug 20181215
				neu1[c] += syn0[c + vocab_hash[randValue] * layer1_size];
			}
			tempValueArray.push_back(randValue);
			cw++;
		}
	}

	else {
		for (d = 0;d < tempValueArray.size();d++) {
			//Debug 20181215
			if (vocab_hash[tempValueArray[d]] <0 || vocab_hash[tempValueArray[d]] >= histogramDimension) continue;
			for (c = 0; c < layer1_size; c++) {
				neu1[c] += syn0[c + vocab_hash[tempValueArray[d]] * layer1_size];
			}
			cw++;
		}
	}

	if (cw <= 0 || maxValue == minValue) {
		tempValueArray.clear();
		neu1.clear();
		neu1e.clear();
		return;
	}
	//#pragma omp parallel for num_threads(thread_num)
	for (c = 0; c < layer1_size; c++)
	{
		neu1[c] /= cw;
	}
	if (hs) {

		for (d = 0; d < vocab[word].codelen; d++) {
			double f = 0;
			long long l2 = vocab[word].point[d] * layer1_size;
			// Propagate hidden -> output
			for (int c = 0; c < layer1_size; c++)
				f += neu1[c] * syn1[c + l2];

			if (f <= -max_exp) continue;
			else if (f >= max_exp) continue;
			else f = expTable[(int)((f + max_exp) * (exp_table_size / max_exp / 2))];
			// 'g' is the gradient multiplied by the learning rate
			//Debug 20181108
			//double g = (1 - vocab[word].code[d] - f) * alpha;
			double g = f * (1 - f) * (vocab[word].code[d] - f) * alpha;

			for (c = 0; c < layer1_size; c++) {
				neu1e[c] += g * syn1[c + l2];
			}
			// Learn weights hidden -> output
			for (c = 0; c < layer1_size; c++) syn1[c + l2] += g * neu1[c];

		}
	}

	if (negative > 0) {
		for (d = 0; d < negative + 1; d++) {
			long long target = 0;
			int label = 0;
			if (d == 0) {
				target = word;
				label = 1;
			}
			else {
				next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
				target = table[(next_random >> 16) % table_size];
				if (target == 0) target = next_random % (vocab_size - 1);
				if (onlyOutOfRangeNegativeUsed) {
					if (vocab[target].intword <= maxValue || vocab[target].intword >= minValue) continue;
				}
				else {
					if (target == word) continue;
				}

				label = 0;
			}
			long long l2 = target * layer1_size;
			real f = 0;
			real g = 0.0f;
			for (c = 0; c < layer1_size; c++) f += neu1[c] * syn1neg[c + l2];
			if (f > max_exp) g = (label - 1) * alpha;
			else if (f < -max_exp) g = (label - 0) * alpha;
			else g = (label - expTable[static_cast<int>((f + max_exp) * (exp_table_size / max_exp / 2))]) * alpha;
			for (c = 0; c < layer1_size; c++) neu1e[c] += g * syn1neg[c + l2];
			for (c = 0; c < layer1_size; c++) syn1neg[c + l2] += g * neu1[c];
		}
	}


	for (d = 0;d < tempValueArray.size();d++) {
		int& tempValue = tempValueArray[d];
		if (vocab_hash[tempValue] <0 || vocab_hash[tempValue] >= histogramDimension) continue;
		for (int c = 0; c < layer1_size; c++)
			syn0[c + vocab_hash[tempValue] * layer1_size] += neu1e[c];
	}
	neu1.clear();
	neu1e.clear();
	tempValueArray.clear();
}

void Volume2Vector::train()
{
	long long id = 0;
	long long a, b, d, cw, word = 0, last_word, sentence_length = 0, sentence_position = 0;
	long long word_count = 0, last_word_count = 0;

	long long l1, l2, c, target, label;
	unsigned long long next_random = (long long)id;
	real f, g;
	//clock_t now;
	long long xDim = dimension.x;
	long long yDim = dimension.y;
	long long zDim = dimension.z;
	long long sentance_size = xDim*yDim*zDim;

	vector<int> sentance;
	sentance.reserve(sentance_size);
	sentance.resize(sentance_size);

	int intword;
	if (word_count - last_word_count > 10000) {
		word_count_actual += word_count - last_word_count;
		last_word_count = word_count;
		if ((debug_mode > 1)) {

			printf("%cAlpha: %f", 13, alpha);
			fflush(stdout);
		}
		alpha = starting_alpha * (1 - word_count_actual / (real)(iter * train_words + 1));
		if (alpha < starting_alpha * 0.0001) alpha = starting_alpha * 0.0001;
	}
	if (sentence_length == 0) {
		//#pragma  omp parallel for
		for (int k = 0;k < zDim;k++) {
			for (int j = 0;j < yDim;j++) {
				for (int i = 0;i < xDim;i++) {

					word = vocab_hash[regularData[k][j][i]];
					//word = readWordIndex(i, j, k, data);
					if (word == -1) {
						sentance[k*xDim*yDim + j*xDim + i] = -1;
						continue;
					}
					word_count++;
					//if (word == 0) break;

					// The subsampling randomly discards frequent words while keeping the ranking same
					if (sample > 0) {
						const real ran = (sqrt(vocab[word].cn / (sample * train_words)) + 1) * (sample * train_words) / vocab[word].cn;
						next_random = next_random * static_cast<unsigned long long>(25214903917) + 11;
						if (ran < (next_random & 0xFFFF) / static_cast<real>(65536)) {
							sentance[k*yDim*xDim + j*xDim + i] = -1;
							continue;
						}
					}

					//sentance记录当前元素在整个vocab中的索引信息
					sentance[k*yDim*xDim + j*xDim + i] = word;

				}
			}
		}
		sentence_position = 0;
	}


	cout << "Debug 20181215 information." << sentance_size << " " << xDim << " " << yDim << " " << zDim << endl;


	vector<vector<int>> histogramBasedVoxel;
	histogramBasedVoxel.resize(histogramDimension);
	for (auto i = 0;i<histogramDimension;i++)
	{
		histogramBasedVoxel[i].reserve(histogramDimension);
		histogramBasedVoxel[i].resize(histogramDimension);
	}

	cout << "alpha:\t" << this->alpha << endl;
	cout << "sample:\t" << this->sample << endl;
	cout << "hs:\t" << this->hs << endl;
	cout << "negative:\t" << this->negative << endl;
	cout << "min_count:\t" << this->min_count << endl;
	cout << "classes:\t" << this->classes << endl;
	cout << "distance:\t" << this->distance2 << endl;
	cout << "histogram_size:\t" << this->histogramDimension << endl;
	cout << "sample_mode:\t" << this->sample_mode << endl;
	cout << "random_sample_number:\t" << this->random_iteration << endl;
	cout << "vector_size:\t" << this->layer1_size << endl;
	cout << "voxelBasedHistogramSaved:\t" << this->isVoxelBasedHistogramSaved << endl;
	cout << "min_count:\t" << this->min_count << endl;


	//#pragma omp parallel for num_threads(thread_num)
	for (long long index = 0; index < sentance_size; index++) {

		cbowGram(index, sentance, distance2, histogramBasedVoxel);


		if (index % 100000 == 0) {
			printf("Degree of completion: %f%%%c", (index*1.0f / sentance_size * 100), 13);
			fflush(stdout);
		}
	}

	if (isVoxelBasedHistogramSaved)
	{
		ofstream histogramFile("./histogram_" + std::to_string(histogramDimension) + ".csv");

		ofstream histogramLogFile("./histogram_log_" + std::to_string(histogramDimension) + ".txt");

		ofstream histogramLogNormalizeforRowFile("./histogram_log_normalize_for_row_" + std::to_string(histogramDimension) + ".txt");

		for (auto i = 0;i<histogramDimension;i++)
		{
			for (auto j = 0;j<histogramDimension - 1;j++)
			{
				histogramFile << histogramBasedVoxel[i][j] << ",";
			}
			histogramFile << histogramBasedVoxel[i][histogramDimension - 1] << endl;
		}
		histogramFile.close();

		histogramLogFile << histogramDimension << endl;
		histogramLogFile << histogramDimension << endl;
		auto max_value = 0.0f;

		vector<vector<float>> histogramLogBasedVoxel;
		histogramLogBasedVoxel.resize(histogramDimension);
		for (auto i = 0;i<histogramDimension;i++)
		{
			histogramLogBasedVoxel[i].reserve(histogramDimension);
			histogramLogBasedVoxel[i].resize(histogramDimension);
		}

		for (auto i = 0;i<histogramDimension;i++)
		{
			for (auto j = 0;j<histogramDimension;j++)
			{
				if (histogramBasedVoxel[i][j] <= 1)
					histogramLogBasedVoxel[i][j] = 0;
				else
					histogramLogBasedVoxel[i][j] = log(histogramBasedVoxel[i][j] * 1.0f);
				max_value = max(max_value, histogramLogBasedVoxel[i][j]);
			}
		}
		if (max_value < 1e-6) max_value = 1.0f;
		//归一化
		for (auto i = 0;i<histogramDimension;i++)
		{
			for (auto j = 0;j<histogramDimension;j++)
			{
				histogramLogFile << histogramLogBasedVoxel[i][j] / max_value << endl;
			}
		}


		histogramLogNormalizeforRowFile << histogramDimension << endl;
		histogramLogNormalizeforRowFile << histogramDimension << endl;
		for (auto i = 0;i<histogramDimension;i++)
		{
			auto max_value1 = 0.0f;
			for (auto j = 0;j<histogramDimension;j++)
			{
				//histogramLogBasedVoxel[i][j] = log(histogramBasedVoxel[i][j] * 1.0f);
				max_value1 = max(max_value1, histogramLogBasedVoxel[i][j]);
			}
			if (max_value1 < 1e-6) max_value1 = 1.0f;
			for (auto j = 0;j<histogramDimension;j++)
			{
				histogramLogNormalizeforRowFile << histogramLogBasedVoxel[i][j] / max_value1 << endl;
			}
		}
		cout << "Histogram based voxels has been saved." << endl;
	}

	printf("\n");
	printf("Vocab size: %lld\n", vocab_size);
	printf("Words in train file: %lld\n", train_words);
	printf("sucess train over!");
	//printf("Train time: %lf\n", (clock() - now) / 1000.0f);
}

void Volume2Vector::save()
{
	{
		const string output_file = "word_vector.txt";
		ofstream fo(output_file);
		
		fo << vocab_size << " " << layer1_size << endl;
		for (auto a = 0; a < vocab_size; a++)
		{
			fo << vocab[a].intword << " ";

			for (auto b = 0; b < layer1_size; b++)
			{
				fo << syn0[a * layer1_size + b] << " ";
			}
			fo << endl;
		}
		fo.close();
		cout << "Save word vector file end..." << endl;
	}
	{
		const string output_file = "cluster.txt";
		ofstream fo(output_file);
		// Run K-means on the word vectors
		int clcn = classes, iter = 10;
		vector<int> centcn;
		centcn.reserve(clcn);
		centcn.resize(clcn);

		vector<int> cl;
		cl.reserve(vocab_size);
		cl.resize(vocab_size);
		real closev, x;
		vector<real> cent;
		cent.reserve(clcn * layer1_size);
		cent.resize(clcn * layer1_size);

		int a, b, c, d;
		for (a = 0; a < vocab_size; a++) cl[a] = a % clcn;
		for (a = 0; a < iter; a++)
		{
			for (b = 0; b < clcn * layer1_size; b++) cent[b] = 0;
			for (b = 0; b < clcn; b++) centcn[b] = 1;
			for (c = 0; c < vocab_size; c++)
			{
				for (d = 0; d < layer1_size; d++) cent[layer1_size * cl[c] + d] += syn0[c * layer1_size + d];
				centcn[cl[c]]++;
			}
			for (b = 0; b < clcn; b++)
			{
				closev = 0;
				for (c = 0; c < layer1_size; c++)
				{
					cent[layer1_size * b + c] /= centcn[b];
					closev += cent[layer1_size * b + c] * cent[layer1_size * b + c];
				}
				closev = sqrt(closev);
				for (c = 0; c < layer1_size; c++) cent[layer1_size * b + c] /= closev;
			}
			for (c = 0; c < vocab_size; c++)
			{
				closev = -10;
				auto closeid = 0;
				for (d = 0; d < clcn; d++)
				{
					x = 0;
					for (b = 0; b < layer1_size; b++) x += cent[layer1_size * d + b] * syn0[c * layer1_size + b];
					if (x > closev)
					{
						closev = x;
						closeid = d;
					}
				}
				cl[c] = closeid;
			}
		}
		// Save the K-means classes
		for (a = 0; a < vocab_size; a++)
		{
			fo << vocab[a].intword << " " <<cl[a] << endl;
		}
		fo.close();
		cout << "Save cluster file end..." << endl;
	}
	{
		
		//Save the similarity array of the volume
		string filePath("similarityArray");
		saveSimilarityArray(filePath);
	}
	
	
	
}
/**
* \brief Calculate the similarity between index_a and index_b. If the cos value < 0, we need set it to 0
* \param index_a
* \param index_b
* \return
*/
real Volume2Vector::similarity(const int index_a, const int index_b) {


	if (index_a <0 || index_b <0) return 0.0f;
	int b;

	vector<real> center;

	center.reserve(layer1_size);
	center.resize(layer1_size);
	for (b = 0; b < layer1_size; b++)
		center[b] = syn0[index_a * layer1_size + b];

	vector<real> compare;
	compare.reserve(layer1_size);
	compare.resize(layer1_size);
	for (b = 0; b < layer1_size; b++)
		compare[b] = syn0[index_b * layer1_size + b];

	real dist = 0;
	real m_a = 0, m_b = 0;
	for (int i = 0; i < layer1_size; i++) {
		dist += center[i] * compare[i];
		m_a += center[i] * center[i];
		m_b += compare[i] * compare[i];
	}

	m_a = sqrt(m_a);
	m_b = sqrt(m_b);
	if (dist < 0) return 0.0f;

	if (m_a <= 1e-8 || m_b <= 1e-8) return 0.0f;
	else
		dist = dist / (m_a*m_b);
	return dist;
}
void Volume2Vector::saveSimilarityArray(string& filepath) {

	int a, b;

	vector<vector<real>> similarity_array;
	similarity_array.resize(histogramDimension);
	for (auto i = 0;i < histogramDimension;i++) {
		similarity_array[i].reserve(histogramDimension);
		similarity_array[i].resize(histogramDimension);
	}

	for (a = 0; a < vocab_size; a++) {
		for (b = 0; b < vocab_size; b++) {
			if (vocab[a].intword >= 0 && vocab[a].intword<histogramDimension&& vocab[b].intword >= 0 && vocab[b].intword<histogramDimension)
			{
				similarity_array[vocab[a].intword][vocab[b].intword] = similarity(vocab_hash[vocab[a].intword], vocab_hash[vocab[b].intword]);
			}
		}
	}

	vector<vector<real>> similarity_sample_array;
	similarity_sample_array.resize(16);
	for (auto& i : similarity_sample_array)
	{
		i.reserve(16);
		i.resize(16);
	}

	ofstream similarity_file(filepath.append("_" + std::to_string(histogramDimension) + ".txt"));

	similarity_file << histogramDimension << endl;
	similarity_file << histogramDimension << endl;

	real max_value = 0.0f;
	
	for (a = 0;a < histogramDimension;a++) {
		for (b = 0;b < histogramDimension;b++) {
			similarity_file << similarity_array[a][b] << endl;
		}
	}
	similarity_file.close();

	const int div = histogramDimension / 16;
	for (a = 0;a < histogramDimension;a++) {
		for (b = 0;b < histogramDimension;b++) {
			similarity_sample_array[a / div][b / div] += similarity_array[a][b];
			if (similarity_sample_array[a / div][b / div] > max_value) max_value = similarity_sample_array[a / div][b / div];
		}
	}

	
	ofstream fp2(string(filepath).append("_" + std::to_string(16) + ".txt"));
	fp2 << 16 << endl << 16 << endl;

	for (a = 0;a < 16;a++) {
		for (b = 0;b < 16;b++) {
			similarity_sample_array[a][b] /= max_value;
			fp2 << similarity_sample_array[a][b] << endl;
		}
	}

	cout << "Save similarity array end..." << endl;
	fp2.close();
}
