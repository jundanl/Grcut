#ifndef __BGRAPH_H__
#define __BGRAPH_H__

#include <opencv2\opencv.hpp>
#include "graph.h"
#include <iostream>
#include <limits>
using namespace std;
using namespace cv;

typedef Graph<double, double, double> GraphType;
typedef void(*error_function)(char *);

class buildgraph{
public:
	buildgraph(){}
	~buildgraph(){ delete g; }
	void mincut(Mat &mask, const Mat &img, const Mat &bg, const Mat &fg);//根据图片的前景和背景模型进行最小化分割，结果保存在mask中
private:
	void initparam(const Mat &bg, const Mat &fg,const Mat &img);
	void calbeta(const Mat &img);//计算beta的值
	void calborderweight(const Mat &img);//计算像素点与领域之间边的权重
	double calplainweight(const Vec3b &point, int label);//计算像素点属于前后景的权重
	double calsingleweight(const Mat &model, int label, const Vec3b &point);//计算像素点属于GMM模型某个高斯函数的概率
	double distance(Vec3d p, Vec3d d);//计算点之间在颜色空间的二阶范数
	void formatgraph(const Mat &mask,const Mat &img);//建立图模型
	//void error(char*);


	//能量函数第二项的常参数
	double gamma = 50;
	double lama = gamma * 9;
	double beta = 0;
	int edge = 0;//普通像素点之间的边
	int points=0;//图像中像素点的个数
	//所有像素点领域的权重，8way
	Mat rweight;//横的边
	Mat cweight;//竖的边
	Mat downweight;//正斜对角线
	Mat upweight;//侧斜对角线
	//前景和背景模型的参数
	Mat backg;//背景模型
	Mat foreg;//前景模型
	int components = 5;//GMM模型的高斯函数个数
	//图
	GraphType *g;
	//测试用
	//double test;
	//Mat te;
	//Mat inv;
	//double det;
};

#endif