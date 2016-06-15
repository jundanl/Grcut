#include <opencv2/opencv.hpp>
using namespace cv;
class GMM
{
public:
	static const int componentsCount = 5; //5个高斯模型
	static const int channelsCount = 3; //处理3通道图片
	GMM(Mat& _model);
	
	void init(); //初始化，全部置零

	void add(int c, const Vec3d pixel); //增加一个像素点

	void update(); //更新GMM参数
	

private:
	Mat model;
	int samplesCount;//总像素点数
	//使用指针直接修改传进的参数
	double *coefs;//每个高斯模型的贡献
	double *mean;//每个高斯模型的期望
	double *cov;//每个高斯模型的方差

	double inv[componentsCount][channelsCount][channelsCount];//协方差矩阵的逆
	double det[componentsCount];//协方差行列式

	double sum[componentsCount][channelsCount];//每个集合的采样点数据和
	double product[componentsCount][channelsCount][channelsCount];//每个集合的采样点数据之间的积，用来计算协方差
	double counts[componentsCount];//每个集合的采样点数量
};