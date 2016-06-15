#include <opencv2/opencv.hpp>
using namespace cv;
class GMM
{
public:
	static const int componentsCount = 5; //5����˹ģ��
	static const int channelsCount = 3; //����3ͨ��ͼƬ
	GMM(Mat& _model);
	
	void init(); //��ʼ����ȫ������

	void add(int c, const Vec3d pixel); //����һ�����ص�

	void update(); //����GMM����
	

private:
	Mat model;
	int samplesCount;//�����ص���
	//ʹ��ָ��ֱ���޸Ĵ����Ĳ���
	double *coefs;//ÿ����˹ģ�͵Ĺ���
	double *mean;//ÿ����˹ģ�͵�����
	double *cov;//ÿ����˹ģ�͵ķ���

	double inv[componentsCount][channelsCount][channelsCount];//Э����������
	double det[componentsCount];//Э��������ʽ

	double sum[componentsCount][channelsCount];//ÿ�����ϵĲ��������ݺ�
	double product[componentsCount][channelsCount][channelsCount];//ÿ�����ϵĲ���������֮��Ļ�����������Э����
	double counts[componentsCount];//ÿ�����ϵĲ���������
};