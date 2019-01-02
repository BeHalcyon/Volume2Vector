#pragma once
#include <iostream>
#include <string>
#include <fstream>
#include <vector>
#include "Volume.h"
using namespace std;


template<typename VolumeType>
class MultiVolume: public Volume<VolumeType>
{
public:
	MultiVolume(int histgramDimension);
	virtual ~MultiVolume();
	void setFileAndRes(vector<string> fileName, int x, int y, int z) override;
	void setFileAndRes(vector<string> fileName, int3& resolution) override;
	void loadVolume() override;
	void setVolumeDimension(int dimension) override;

	void deleteData() override;

	bool isInMemory() const override;

	vector < vector < vector<VolumeType>>>& getData() override;
	void calculateRegularVolume() override;
	vector < vector < vector<int>>>& getRegularData() override;
	int3& getDimension() override;
	int getHistogramDimension() const override;

private:
	//vector<VolumeType> max_value_array;
	//vector<VolumeType> min_value_array;
	//vector
};



template<typename VolumeType>
MultiVolume<VolumeType>::MultiVolume(int histgramDimension): Volume<VolumeType>(histgramDimension)
{
	this->isDataInMemory = false;
	this->isRegularDataGenerated = false;
	this->max_value = -10000000;
	this->min_value = 10000000;
	this->histogramDimension = histgramDimension;
}

template<typename VolumeType>
MultiVolume<VolumeType>::~MultiVolume()
{
	//deleteData();
	//~Volume<VolumeType>();
}


template<typename VolumeType>
void MultiVolume<VolumeType>::setFileAndRes(vector<string> fileName, int x, int y, int z)
{
	for(int i=0;i<fileName.size();i++)
	{
		max_value_array.push_back(-10000000);
		min_value_array.push_back(10000000);
	}

	this->volumeFileName = fileName;
	this->volumeRes.x = x;
	this->volumeRes.y = y;
	this->volumeRes.z = z;
}

template<typename VolumeType>
void MultiVolume<VolumeType>::setFileAndRes(vector<string> fileName, int3& resolution)
{
	this->volumeFileName = fileName;
	this->volumeRes = resolution;
}

template<typename VolumeType>
vector < vector < vector<VolumeType>>>& MultiVolume<VolumeType>::getData()
{
	if (!this->isDataInMemory) { loadVolume(); }
	return this->volumeData;
}



template <typename VolumeType>
void MultiVolume<VolumeType>::calculateRegularVolume()
{
	loadVolume();

	this->regularData.resize(this->volumeRes.z);
	for (auto x = 0; x < this->volumeRes.z; ++x)
	{
		this->regularData[x].resize(this->volumeRes.y);
		for (auto y = 0; y < this->volumeRes.y; ++y)
		{
			this->regularData[x][y].resize(this->volumeRes.x);
			for (auto z = 0; z < this->volumeRes.x; ++z)
			{
				this->regularData[x][y][z] = ((this->volumeData[x][y][z] - this->min_value)*1.0f / ((this->max_value - this->min_value)*1.0f))*(this->histogramDimension - 1);
			}
		}
	}
	cout << "Calculate regular volume successfully." << endl;
	//cout << "Max_value and min_value are: " << max_value << " " << min_value << endl;
	this->isRegularDataGenerated = true;
}

template <typename VolumeType>
vector<vector<vector<int>>>& MultiVolume<VolumeType>::getRegularData()
{
	if (!this->isRegularDataGenerated) { calculateRegularVolume(); }
	return this->regularData;
}

template <typename VolumeType>
int3& MultiVolume<VolumeType>::getDimension()
{
	return this->volumeRes;
}

template <typename VolumeType>
int MultiVolume<VolumeType>::getHistogramDimension() const
{
	return this->histogramDimension;
}


/**
* \brief Load one volume to memory
*/
template<typename VolumeType>
void MultiVolume<VolumeType>::loadVolume()
{
	if (this->isDataInMemory)
	{
		return;
	}

	std::ifstream in(this->volumeFileName, ios::in | ios::binary);
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


		this->volumeData.resize(this->volumeRes.z);
		for (auto x = 0; x < this->volumeRes.z; ++x)
		{
			this->volumeData[x].resize(this->volumeRes.y);
			for (auto y = 0; y < this->volumeRes.y; ++y)
			{
				this->volumeData[x][y].reserve(this->volumeRes.x);
				this->volumeData[x][y].resize(this->volumeRes.x);
				for (auto z = 0; z <this->volumeRes.x; ++z)
				{
					int src_idx = sizeof(VolumeType) * (z + y * this->volumeRes.x + x * this->volumeRes.x* this->volumeRes.y);
					memcpy(&this->volumeData[x][y][z], &contents[src_idx], sizeof(VolumeType));
					this->max_value = this->volumeData[x][y][z]> this->max_value ? this->volumeData[x][y][z] : this->max_value;
					this->min_value = this->volumeData[x][y][z]< this->min_value ? this->volumeData[x][y][z] : this->min_value;
				}
			}
		}
		cout << "Translate data into volume type successfully." << endl;
		//cout << "Max_value and min_value are: " << static_cast<int>(max_value) << " " << static_cast<int>(min_value) << endl;
	}
	free(contents);
}

template <typename VolumeType>
void MultiVolume<VolumeType>::setVolumeDimension(int dimension)
{
	this->histogramDimension = dimension;
}

template<typename VolumeType>
void MultiVolume<VolumeType>::deleteData()
{
	if (this->isDataInMemory)
	{
		this->volumeData.clear();
		//volumeData = nullptr;
		this->isDataInMemory = false;
		this->regularData.clear();
		this->isRegularDataGenerated = false;
	}
}

template<typename VolumeType>
bool MultiVolume<VolumeType>::isInMemory() const
{
	return this->isDataInMemory;
}
