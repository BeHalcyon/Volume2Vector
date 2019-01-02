#pragma once
#include <vector>
#include <iostream>
using namespace std;
#define MAX_CODE_LENGTH 100

class Neuron
{
public:
	Neuron();
	~Neuron();

	long long cn = 0;
	int point[MAX_CODE_LENGTH];
	char code[MAX_CODE_LENGTH];
	char codelen = 0;//code表示哈夫曼编码，codelen为编码长度
	int intword = -1;

	bool operator<(Neuron& b) const
	{
		return cn > b.cn;
	}

};

