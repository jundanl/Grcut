#include <opencv2/opencv.hpp>
using namespace cv;
class GMM
{
public:
	static const int componentsCount = 5; //5个高斯模型
	static const int channelsCount = 3; //处理3通道图片
	static const int modelSize = 1 + channelsCount + channelsCount * channelsCount;//对象个数
	static const int dataCount = componentsCount * modelSize;//总参数个数

	GMM(Mat& _model); //使用外部数据构造
	GMM();//默认构造，数据全置零

	void copyFrom(Mat& _model); //从外部model拷贝数据
	void copyTo(Mat& _model); //拷贝数据到外部model

	void checkModel(Mat& _model);//检测传入model是否初始化
	
	void init(); //更新初始化，全部置零

	void add(int c, const Vec3d pixel); //增加一个像素点到第c个高斯模型

	void update(); //更新GMM参数
	
	//根据img和mask生成GMM，并拷贝一份数据到model
	static void initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, Mat& bgdModel, Mat& fgdModel);

private:
	int samplesCount;//总像素点数
	double data[dataCount];//储存所有数据
	//指针指向位置
	double *coefs;//每个高斯模型的贡献
	double *mean;//每个高斯模型的期望
	double *cov;//每个高斯模型的方差

	double inv[componentsCount][channelsCount][channelsCount];//协方差矩阵的逆
	double det[componentsCount];//协方差行列式

	double sum[componentsCount][channelsCount];//每个集合的采样点数据和
	double product[componentsCount][channelsCount][channelsCount];//每个集合的采样点数据之间的积，用来计算协方差
	double counts[componentsCount];//每个集合的采样点数量

};