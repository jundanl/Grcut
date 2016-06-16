#ifndef __KMEANS__
#define __KMEANS__
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <time.h>
#include <string.h>
#include <stdlib.h>
#define abs(a) ((a)>0?(a):(-(a)))
using namespace cv;

/*����uchar��ά�����ڴ�*/
uchar** malloc_uchar2D(int rows, int cols) {
	int size = sizeof(uchar);
	int point_size = sizeof(uchar*);
	uchar **arr = (uchar**)malloc(point_size * rows + size * rows * cols);
	if (arr != NULL) {
		memset(arr, 0, point_size * rows + size * rows * cols);
		uchar* head = (uchar*)((uchar**)arr +  rows);
		while (rows--) {
			arr[rows] = (uchar*)((uchar*)head + rows * cols);
		}
		return arr;
	}
	return NULL;
	/*
	int i;
	uchar **arr = (uchar**) malloc(point_size * rows);
	for (i = 0;i < rows;i++) {
		arr[i] = (uchar*)malloc(size * cols);
	}
	return arr;
	*/
}

/*����double��ά�����ڴ�*/
double** malloc_double2D(int rows, int cols) {
	int size = sizeof(double);
	int point_size = sizeof(double*);
	double **arr = (double**)malloc(point_size * rows + size * rows * cols);
	if (arr != NULL) {
		memset(arr, 0, point_size * rows + size * rows * cols);
		double* head = (double*)((double**)arr +  rows);
		while (rows--) {
			arr[rows] = (double*)((double*)head + rows * cols);
		}
		return arr;
	}
	return NULL;
}

/*
void free_uchar2D(uchar** arr, int rows) {
	free(arr);
}

void free_double2D(double **arr, int rows) {
	free(arr);
}
*/

//���ѡȡknum������Ϊ��ʼ�����ĵ�
void kmInitialize(double **kp, uchar **p, const int knum, const int num, const int dimension) {
	int count = 0;
	srand(time(0));
	int *flag = new int[num];
	memset(flag, 0, sizeof(int) * num);
	while (count < knum) {
		int k = rand() % num;
		if (flag[k] == 0) {
			for (int i = 0;i < dimension;i++) 
				kp[count][i] = p[k][i];
			count++;
			flag[k] = 1;
		}
	}
	delete[] flag;
	/*
	for (i = 0;i < knum;i++) {
		int k;
		for (k = 0;k < dimension;k++) {
			kp[i][k] = p[num-i-1][k];
		}
	}*/
}

//��������
int kmUpdate(double **kp, uchar **p, int knum, int num, int dimension, int *belong) {
	int i, j, k;
	int flag = 0, count = 0;
	double distance, mindistance;
	int minindex;
	int *cnt = new int[knum];
	double **tmp = malloc_double2D(knum, dimension);
	int *prebelong = new int[num];
	memcpy(prebelong, belong, sizeof(int) * num);
	memset(belong, -1, num * sizeof(int));
	memset(cnt, 0, sizeof(int) * knum);
	//�������ڵ����ĵ�ֵ
	for (i = 0;i < knum;i++) {
		for (j = 0;j < dimension;j++) {
			tmp[i][j] = kp[i][j];
			kp[i][j] = 0;
		}
	}
	//Ϊÿ��������µ����ĵ�
	for (i = 0;i < num;i++) {
		minindex = 0;
		mindistance = 1e19;
		//�ҵ���������ĵ�
		for (j = 0;j < knum;j++) {
			distance = 0;
			for (k = 0;k < dimension;k++){
				distance += ((double)(p[i][k]) - tmp[j][k]) * ((double)(p[i][k]) - tmp[j][k]);
			}
			if (distance < mindistance) {
				mindistance = distance;
				minindex = j;
			}
		}
		belong[i] = minindex;
		cnt[minindex]++;
		//�����������ĵ�����
		for (k = 0;k < dimension;k++)
			kp[minindex][k] += p[i][k];
		if (belong[i] != prebelong[i]) {
			flag = 1;
			count++;
		}
	}
	printf("%d\n", count);
	double maxdiff = 0;
	double mindiff = std::numeric_limits<double>::epsilon();
	//����ÿ�����ĵ��λ��
	for (i = 0;i < knum;i++) {
		for (j = 0;j < dimension;j++) {
			if (cnt[i] != 0) {
				kp[i][j] = kp[i][j] / (double)cnt[i]; 
				//�������ֵ
				if (abs(kp[i][j] - tmp[i][j]) > maxdiff) {
					maxdiff = abs(kp[i][j] - tmp[i][j]);
				}
				if (abs(kp[i][j] - tmp[i][j]) < maxdiff) {
					mindiff = abs(kp[i][j] - tmp[i][j]);
				}
			}
		}
	}
	printf("%.9lf\n", maxdiff);
	printf("%.9lf\n", mindiff);
	free(tmp);
	delete[] cnt;
	delete[] prebelong;
	return flag;
}

// mode=0;res���Ϊÿ�����ж��ٸ��㣬 mode=1:res���Ϊÿ���������ĸ���
void kmCheck(double **kp, uchar **p, int knum, int num, int dimension, int *res, int mode) {
	int i, j, k;
	for (i = 0;i < knum;i++) res[i] = 0;
	for (i = 0;i < num;i++) {
		double mindistance = 0x7fffffff;
		int minindex = 0;
		//Ѱ����������ĵ�
		for (j = 0;j < knum;j++) {
			double distance = 0;
			for (k = 0;k < dimension;k++) {
				distance += ((double)(p[i][k]) - kp[j][k]) * ((double)(p[i][k]) - kp[j][k]);
			}
			if (distance < mindistance) {
				mindistance = distance;
				minindex = j;
			}
		}
		if (mode == 0) {
			res[minindex]++;
		} else {
			res[i] = minindex;
		}
	}
}

void k_means(vector<Vec3f> &points, Mat& labels, int maxIt, int knum, int dimension) {
	uchar **p = malloc_uchar2D(points.size(), dimension);
	double **kp = malloc_double2D(knum, dimension);
	if (labels.empty()) {
		labels.create(points.size(), 1, CV_32SC1);
	}
	int *belong = labels.ptr<int>(0);
	for (int i = 0; i < points.size(); i++)
		for (int j = 0; j < dimension; j++)
		{
			p[i][j] = points[i][j];
		}
	kmInitialize(kp, p, knum, points.size(), dimension);
	for (int i = 0; i < maxIt; i++)
	{
		if (!kmUpdate(kp, p, knum, points.size(), dimension, belong))
			break;
	}
	free(p);
	free(kp);
}

#endif