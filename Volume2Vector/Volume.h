#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
//#include <Qtcore\QFile>
//#include <QtCore\QDataStream>
//#include <Qtcore\Qstring>
//#include "define.h"
using namespace std;

class int3
{
public:
	int x=0;
	int y=0;
	int z=0;
public:
	int3(int x=0, int y = 0, int z = 0):
	x(x),y(y),z(z)
	{
		
	}
};

class double3
{
public:
	double x = 0;
	double y = 0;
	double z = 0;
public:
	double3(double x = 0, double y = 0, double z = 0) :
		x(x), y(y), z(z)
	{

	}
};

template<typename VolumeType>
class Volume
{
public:
	Volume(int histgramDimension);
	virtual ~Volume();
	virtual void setFileAndRes(vector<string> fileName, int x, int y, int z);
	virtual void setFileAndRes(vector<string> fileName, int3& resolution);
	virtual void loadVolume();
	virtual void setVolumeDimension(int dimension);

	virtual void deleteData();

	virtual bool isInMemory() const;

	virtual vector<vector<vector<vector<VolumeType>>>>& getData();
	virtual void calculateRegularVolume();
	virtual vector < vector < vector<int>>>& getRegularData();
	virtual int3& getDimension();
	virtual int getHistogramDimension() const;


protected:
	bool isDataInMemory;
	bool isRegularDataGenerated;
	vector<double> max_value, min_value;
	int histogramDimension = 256;

	int volume_number = 0;

	int3 volumeRes;
	vector<string> volumeFileName;

	vector<vector<vector<vector<VolumeType>>>> volumeData; //The first dimension means the id of current volume;
	vector<vector<vector<int>>> regularData; // Merge multiple volume to one

};



template<typename VolumeType>
Volume<VolumeType>::Volume(int histgramDimension): volumeRes(0, 0, 0)
{
	isDataInMemory = false;
	isRegularDataGenerated = false;
	//max_value = -10000000;
	//min_value = 10000000;
	histogramDimension = histgramDimension;
}

template<typename VolumeType>
Volume<VolumeType>::~Volume()
{
	deleteData();
}


template<typename VolumeType>
void Volume<VolumeType>::setFileAndRes(vector<string> fileName, int x, int y, int z)
{
	volumeFileName = fileName;
	volume_number = fileName.size();
	for(int i=0;i<volume_number;i++)
	{
		max_value.push_back(-10000000);
		min_value.push_back(10000000);
	}
	volumeRes.x = x;
	volumeRes.y = y;
	volumeRes.z = z;
}

template<typename VolumeType>
void Volume<VolumeType>::setFileAndRes(vector<string> fileName, int3& resolution)
{
	volumeFileName = fileName;
	volume_number = fileName.size();
	for (int i = 0;i<volume_number;i++)
	{
		max_value.push_back(-10000000);
		min_value.push_back(10000000);
	}
	volumeRes = resolution;
}

template<typename VolumeType>
vector<vector<vector<vector<VolumeType>>>>& Volume<VolumeType>::getData()
{
	if (!isDataInMemory) { loadVolume(); }
	return volumeData;
}



template <typename VolumeType>
void Volume<VolumeType>::calculateRegularVolume()
{
	loadVolume();
	const auto local_dimension = static_cast<int>(pow(histogramDimension, static_cast<double>(1.0f) / volume_number));
	cout << local_dimension << endl;
	int regular_min_value = histogramDimension * 100;
	int regular_max_value = -histogramDimension * 100;
	regularData.resize(volumeRes.z);
	for (auto x = 0; x < volumeRes.z; ++x)
	{
		regularData[x].resize(volumeRes.y);
		for (auto y = 0; y < volumeRes.y; ++y)
		{
			regularData[x][y].resize(volumeRes.x);
			for (auto z = 0; z < volumeRes.x; ++z)
			{
				vector<int> volume_value;
				for(auto i=0;i<volume_number;i++)
				{
					volume_value.push_back(((volumeData[i][x][y][z] - min_value[i])*1.0f / ((max_value[i] - min_value[i])*1.0f))*(local_dimension - 1));
					//cout << volume_value[i] << " ";
				}
				//cout << endl;

				auto result_value = 0;
				for(auto i=0;i<volume_number;i++)
				{
					result_value += pow(local_dimension, volume_number - i-1) * volume_value[i];

				}
				regularData[x][y][z] = result_value;
				regular_max_value = max(regularData[x][y][z], regular_max_value);
				regular_min_value = min(regularData[x][y][z], regular_min_value);

			}
		}
	}
	cout << "Calculate regular volume successfully." << endl;
	cout << "Max regular value : "<<regular_max_value<<" min regular value : "<< regular_min_value << endl;
	//cout << "Max_value and min_value are: " << max_value << " " << min_value << endl;
	isRegularDataGenerated = true;
}

template <typename VolumeType>
vector<vector<vector<int>>>& Volume<VolumeType>::getRegularData()
{
	if (!isRegularDataGenerated) { calculateRegularVolume(); }
	return regularData;
}

template <typename VolumeType>
int3& Volume<VolumeType>::getDimension()
{
	return this->volumeRes;
}

template <typename VolumeType>
int Volume<VolumeType>::getHistogramDimension() const
{
	return this->histogramDimension;
}


/**
 * \brief Load one volume to memory
 */
template<typename VolumeType>
void Volume<VolumeType>::loadVolume()
{
	if (isDataInMemory)
	{
		return;
	}
	volumeData.resize(volume_number);
	for(auto i=0;i<volume_number;i++)
	{
		std::ifstream in(volumeFileName[i], ios::in | ios::binary);
		unsigned char *contents = nullptr;
		if (in)
		{
			in.seekg(0, std::ios::end);
			const long int fileSize = in.tellg();
			contents = static_cast<unsigned char*>(malloc(static_cast<size_t>(fileSize + 1)));
			in.seekg(0, std::ios::beg);
			in.read(reinterpret_cast<char*>(contents), fileSize);
			in.close();
			contents[fileSize] = '\0';
			cout << "Load data successfully. The char size of the file is : " << fileSize << endl;


			volumeData[i].resize(volumeRes.z);
			for (auto x = 0; x < volumeRes.z; ++x)
			{
				volumeData[i][x].resize(volumeRes.y);
				for (auto y = 0; y < volumeRes.y; ++y)
				{
					volumeData[i][x][y].reserve(volumeRes.x);
					volumeData[i][x][y].resize(volumeRes.x);
					for (auto z = 0; z < volumeRes.x; ++z)
					{
						int src_idx = sizeof(VolumeType) * (z + y * volumeRes.x + x * volumeRes.x* volumeRes.y);
						memcpy(&volumeData[i][x][y][z], &contents[src_idx], sizeof(VolumeType));
						max_value[i] = volumeData[i][x][y][z]> max_value[i] ? volumeData[i][x][y][z] : max_value[i];
						min_value[i] = volumeData[i][x][y][z]< min_value[i] ? volumeData[i][x][y][z] : min_value[i];
					}
				}
			}
			//cout << "Translate data " << i << " into volume type successfully." << endl;
			cout << "Max value : "<<max_value[i]<<" min_value : "<<min_value[i] << endl;
			//cout << "Max_value and min_value are: " << static_cast<int>(max_value) << " " << static_cast<int>(min_value) << endl;
		}
		free(contents);
	}
}

template <typename VolumeType>
void Volume<VolumeType>::setVolumeDimension(int dimension)
{
	this->histogramDimension = dimension;
}

template<typename VolumeType>
void Volume<VolumeType>::deleteData()
{
	if (isDataInMemory)
	{
		volumeData.clear();
		//volumeData = nullptr;
		isDataInMemory = false;
		regularData.clear();
		isRegularDataGenerated = false;
	}
}

template<typename VolumeType>
bool Volume<VolumeType>::isInMemory() const
{
	return isDataInMemory;
}
