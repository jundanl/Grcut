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
const int nstep = 8;
const int nx[8] = { 0, 1, 0, -1, -1, -1, 1, 1 };
const int ny[8] = { 1, 0, -1, 0, -1, 1, -1, 1 };

const int rstep = 4;
const int rx[4] = {0,1,0,-1};
const int ry[4] = {1,0,-1,0};

const double MAXNUM = 9999999;
const int COE = 10000;

class BorderMatting
{
public:
	BorderMatting();
	~BorderMatting();
	void borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask);
private:
	void ParameterizationContour(const Mat& edge, Contour& contour);
	void dfs(int x, int y, const Mat& mask, Mat& amask, Contour& contour);
	void StripInit(const Mat& mask, Contour& contour, Strip& strip);
	void EnergyMinimization(const Mat& oriImg, const Mat& mask, Contour& contour, Strip& strip);
	inline double Vfunc(double ddelta, double dsigma)
	{
		return lamda1*pow(ddelta, 2.0) + lamda2*pow(dsigma, 2.0);
	}
	void init(const Mat& img);
	double Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, Strip& strip, const Mat& gray);
	void CalculateMask(Mat& bordermask, const Mat& mask);
	const int sigmaLevels = 10;
	const int deltaLevels = 10;
	const double sigma = 0.5;
	const double delta = 0.3;
	const int lamda1 = 50;
	const int lamda2 = 1000;
	int sections = 0;
	int rows, cols;
	int areacnt;
	int tot;
	Contour contour;
	Strip strip;
	double ef[5000][30][10];
	dands rec[5000][30][10];
	vector<dands> vecds;
};

