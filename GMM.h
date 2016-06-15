#include <opencv2/opencv.hpp>
using namespace cv;
class GMM
{
public:
	static const int componentsCount = 5; //5����˹ģ��
	static const int channelsCount = 3; //����3ͨ��ͼƬ
	static const int modelSize = 1 + channelsCount + channelsCount * channelsCount;//�������
	static const int dataCount = componentsCount * modelSize;//�ܲ�������

	GMM(Mat& _model); //ʹ���ⲿ���ݹ���
	GMM();//Ĭ�Ϲ��죬����ȫ����

	void copyFrom(Mat& _model); //���ⲿmodel��������
	void copyTo(Mat& _model); //�������ݵ��ⲿmodel

	void checkModel(Mat& _model);//��⴫��model�Ƿ��ʼ��
	
	void init(); //���³�ʼ����ȫ������

	void add(int c, const Vec3d pixel); //����һ�����ص㵽��c����˹ģ��

	void update(); //����GMM����
	
	//����img��mask����GMM��������һ�����ݵ�model
	static void initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, Mat& bgdModel, Mat& fgdModel);

private:
	int samplesCount;//�����ص���
	double data[dataCount];//������������
	//ָ��ָ��λ��
	double *coefs;//ÿ����˹ģ�͵Ĺ���
	double *mean;//ÿ����˹ģ�͵�����
	double *cov;//ÿ����˹ģ�͵ķ���

	double inv[componentsCount][channelsCount][channelsCount];//Э����������
	double det[componentsCount];//Э��������ʽ

	double sum[componentsCount][channelsCount];//ÿ�����ϵĲ��������ݺ�
	double product[componentsCount][channelsCount][channelsCount];//ÿ�����ϵĲ���������֮��Ļ�����������Э����
	double counts[componentsCount];//ÿ�����ϵĲ���������

};