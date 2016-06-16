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
	void mincut(Mat &mask, const Mat &img, const Mat &bg, const Mat &fg);//����ͼƬ��ǰ���ͱ���ģ�ͽ�����С���ָ���������mask��
private:
	void initparam(const Mat &bg, const Mat &fg,const Mat &img);
	void calbeta(const Mat &img);//����beta��ֵ
	void calborderweight(const Mat &img);//�������ص�������֮��ߵ�Ȩ��
	double calplainweight(const Vec3b &point, int label);//�������ص�����ǰ�󾰵�Ȩ��
	double calsingleweight(const Mat &model, int label, const Vec3b &point);//�������ص�����GMMģ��ĳ����˹�����ĸ���
	double distance(Vec3d p, Vec3d d);//�����֮������ɫ�ռ�Ķ��׷���
	void formatgraph(const Mat &mask,const Mat &img);//����ͼģ��
	//void error(char*);


	//���������ڶ���ĳ�����
	double gamma = 50;
	double lama = gamma * 9;
	double beta = 0;
	int edge = 0;//��ͨ���ص�֮��ı�
	int points=0;//ͼ�������ص�ĸ���
	//�������ص������Ȩ�أ�8way
	Mat rweight;//��ı�
	Mat cweight;//���ı�
	Mat downweight;//��б�Խ���
	Mat upweight;//��б�Խ���
	//ǰ���ͱ���ģ�͵Ĳ���
	Mat backg;//����ģ��
	Mat foreg;//ǰ��ģ��
	int components = 5;//GMMģ�͵ĸ�˹��������
	//ͼ
	GraphType *g;
	//������
	//double test;
	//Mat te;
	//Mat inv;
	//double det;
};

#endif