#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "BorderMatting.h"
#include <vector>
#include <unordered_map>
#include <math.h>
using namespace std;
using namespace cv;

struct point{
	int x;
	int y;
};

struct para_point{
	point p;
	int index;
	int section;
	double delta;
	double sigma;
};

struct inf_point{
	point p;
	int dis;
	int area;
};

struct dands{
	int delta;
	int sigma;
};

typedef vector<double[30][10]> Energyfunction;
typedef vector<dands[30][10]> Record;
typedef vector<para_point> Contour;
typedef unordered_map<int, inf_point> Strip;

/*��������Ϊ���ڵ�8����*/
#define nstep 8
const int nx[nstep] = { 0, 1, 0, -1, -1, -1, 1, 1 };
const int ny[nstep] = { 1, 0, -1, 0, -1, 1, -1, 1 };

#define COE 10000

#define stripwidth 6

#define L 20

/*ŷʽ����Ϊ1�����ڵ�*/
#define rstep 4
const int rx[rstep] = {0,1,0,-1};
const int ry[rstep] = {1,0,-1,0};

#define MAXNUM 9999999;

#define sigmaLevels  15
#define deltaLevels  11

class BorderMatting
{
public:
	BorderMatting();
	~BorderMatting();
	void borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask);
private:
	void ParameterizationContour(const Mat& edge);
	void dfs(int x, int y, const Mat& mask, Mat& amask);
	void StripInit(const Mat& mask);
	void EnergyMinimization(const Mat& oriImg, const Mat& mask);
	inline double Vfunc(double ddelta, double dsigma)
	{
		return (lamda1*pow(ddelta, 2.0) + lamda2*pow(dsigma, 2.0))/200;
	}
	void init(const Mat& img);
	double Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray);
	void CalculateMask(Mat& bordermask, const Mat& mask);
	void display(const Mat& oriImg,const Mat& mask);

	const int lamda1 = 50;
	const int lamda2 = 1000;
	int sections; //������������
	int rows, cols; 
	int areacnt; //����������������ϵ�ĸ�����
	int tot;
	Contour contour; //����
	Strip strip; //����
	double ef[5000][deltaLevels][sigmaLevels];
	dands rec[5000][deltaLevels][sigmaLevels];
	vector<dands> vecds;
};

