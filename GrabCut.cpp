#include "GrabCut.h"

class GMM
{
public:
	static const int componentsCount = 5; //5����˹ģ��
	static const int channelsCount = 3; //����3ͨ��ͼƬ
	GMM(Mat& _model)
	{
		const int modelSize = 1 + channelsCount + channelsCount * channelsCount;
		if (_model.empty()) {
			_model.create(1, modelSize * componentsCount, CV_64FC1);
			_model.setTo(Scalar(0));
		}
		else if (_model.type() != CV_64FC1 || _model.rows != 1 || _model.cols != modelSize * componentsCount)
			CV_Error(CV_StsBadArg, "_model must be in 1 * 13 double type.");
		model = _model;

		//����˳�� 1��Ȩ�أ�3����ֵ��9��Э����
		coefs = model.ptr<double>(0);
		mean = coefs + componentsCount;
		cov = mean + channelsCount * componentsCount;
	}

	//��ʼ����ȫ������
	void init() 
	{
		memset(sum, 0, sizeof(double) * componentsCount * channelsCount);
		memset(product, 0, sizeof(double) * componentsCount * channelsCount * channelsCount);
		memset(counts, 0, sizeof(double) * componentsCount);
		samplesCount = 0;
	}

	//����һ�����ص�
	void add(int c, const Vec3d pixel)
	{
		for (int i = 0; i < channelsCount; i++) 
			sum[c][i] += pixel[i];

		for (int i = 0; i < channelsCount; i++)
			for (int j = 0; j < channelsCount;j++) 
			{
				product[c][i][j] += pixel[i] * pixel[j];
			}
		counts[c]++;
		samplesCount++;
	}

	//����GMM����
	void update() 
	{
		for (int c = 0; c < componentsCount; c++)
		{
			if (counts[c] == 0) {
				coefs[c] = 0;
				continue;
			}
			//��c����˹ģ�͵Ĺ���
			coefs[c] = (double)counts[c] / (double)samplesCount;

			//�����c����˹ģ�͵�����
			int meanBase = c * channelsCount;
			for (int i = 0; i < channelsCount; i++)
			{
				mean[meanBase + i] = sum[c][i] / counts[c];
			}

			//�����c����˹ģ�͵�Э����
			double *co = cov + c * channelsCount * channelsCount;
			for (int i = 0; i < channelsCount; i++)
				for (int j = 0; j < channelsCount; j++)
				{
					co[i * channelsCount + j] =
						product[c][i][j] / counts[c] - mean[meanBase + i] * mean[meanBase + j];

				}
			
			//�����c����˹ģ�͵�Э��������������ʽ
			det[c] = co[0] * (co[4] * co[8] - co[5] * co[7])
				- co[1] * (co[3] * co[8] - co[5] * co[6])
				+ co[2] * (co[3] * co[7] - co[4] * co[6]);
			//Ϊ�˼��������Ϊ����ʽС�ڵ���0�ľ���Խ�����������
			if (det[c] <= std::numeric_limits<double>::epsilon())
			{
				co[0] += 0.001;
				co[4] += 0.001;
				co[8] += 0.001;

				det[c] = co[0] * (co[4] * co[8] - co[5] * co[7])
					- co[1] * (co[3] * co[8] - co[5] * co[6])
					+ co[2] * (co[3] * co[7] - co[4] * co[6]);
			}
			CV_Assert(det[c] > std::numeric_limits<double>::epsilon());
			
			//������
			inv[c][0][0] =  (co[4]*co[8] - co[5]*co[7]) / det[c];  
	        inv[c][1][0] = -(co[3]*co[8] - co[5]*co[6]) / det[c];  
	        inv[c][2][0] =  (co[3]*co[7] - co[4]*co[6]) / det[c];  
	        inv[c][0][1] = -(co[1]*co[8] - co[2]*co[7]) / det[c];  
	        inv[c][1][1] =  (co[0]*co[8] - co[2]*co[6]) / det[c];  
	        inv[c][2][1] = -(co[0]*co[7] - co[1]*co[6]) / det[c];  
	        inv[c][0][2] =  (co[1]*co[5] - co[2]*co[4]) / det[c];  
	        inv[c][1][2] = -(co[0]*co[5] - co[2]*co[3]) / det[c];  
	        inv[c][2][2] =  (co[0]*co[4] - co[1]*co[3]) / det[c];  
		}

	}

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


GrabCut2D::~GrabCut2D(void)
{
}

//Using rect initialize the pixel 
void setRectInMask(const Mat& img,Mat& mask, Rect& rect)
{
	assert(!mask.empty());
	mask.setTo(GC_BGD);   //GC_BGD == 0
	rect.x = max(0, rect.x);
	rect.y = max(0, rect.y);
	rect.width = min(rect.width, img.cols - rect.x);
	rect.height = min(rect.height, img.rows - rect.y);
	(mask(rect)).setTo(Scalar(GC_PR_FGD));    //GC_PR_FGD == 3 
}

void initGMM(const Mat& img, GMM& bgdGMM, GMM& fgdGMM, const Mat& mask)
{
	/*----------------Kmeans ��������-----------------*/
	const int kmeansItCount = 10; //kmeans����������
	const int kmeasType = KMEANS_RANDOM_CENTERS; //kmeans��ʼ����ѡȡ����
	const float kmeansEpsilon = 0.0; //kmeans��ֹ���������

	Mat bgdLabels, fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;

	//���ݿ�ѡ���䵽����GMM
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (mask.at<uchar>(i, j) == GC_BGD || mask.at<uchar>(i, j) == GC_PR_BGD)
				bgdSamples.push_back(img.at<Vec3b>(i, j));
			else
				fgdSamples.push_back(img.at<Vec3b>(i, j));
		}

	//����Kmeans�������
	CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
	Mat _bgdSamples(bgdSamples.size(), GMM::channelsCount, CV_32FC1, &bgdSamples[0][0]);
	Mat _fgdSamples(fgdSamples.size(), GMM::channelsCount, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::componentsCount, bgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);
	kmeans(_fgdSamples, GMM::componentsCount, fgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);

	/*-------------���þ���������GMM����-------------*/
	//��ʼ��
	bgdGMM.init();
	fgdGMM.init();
	
	//���ݾ����ǩΪÿ����˹ģ�ͷ��������
	for (int i = 0; i < bgdSamples.size(); i++)
	{
		bgdGMM.add(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	}
	for (int i = 0; i < fgdSamples.size(); i++)
	{
		fgdGMM.add(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	}

	//�������
	bgdGMM.update();
	fgdGMM.update();

}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;

	const Mat& img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();

	GMM bgdGMM(bgdModel);
	GMM fgdGMM(fgdModel);

	/*-------------����ǰ��mask---------------*/
	if (mode == GC_WITH_RECT)
	{
		cout << "GC_WITH_RECT" << endl;
		setRectInMask(img, mask, rect);
	}

	/*------------���ݿ�ѡ����GMM-------------*/
	initGMM(img, bgdGMM, fgdGMM, mask);

	if ((mode != GC_CUT) || (iterCount == 0))
		return;


//һ.�������ͣ�
	//���룺
	 //cv::InputArray _img,     :�����colorͼ��(����-cv:Mat)
     //cv::Rect rect            :��ͼ���ϻ��ľ��ο�����-cv:Rect) 
  	//int iterCount :           :ÿ�ηָ�ĵ�������������-int)


	//�м����
	//cv::InputOutputArray _bgdModel ��   ����ģ�ͣ��Ƽ�GMM)������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�
	//cv::InputOutputArray _fgdModel :    ǰ��ģ�ͣ��Ƽ�GMM) ������-13*n�������������double���͵��Զ������ݽṹ������Ϊcv:Mat������Vector/List/����ȣ�


	//���:
	//cv::InputOutputArray _mask  : ����ķָ��� (���ͣ� cv::Mat)

//��. α�������̣�
	//1.Load Input Image: ����������ɫͼ��;
	//2.Init Mask: �þ��ο��ʼ��Mask��Labelֵ��ȷ��������0�� ȷ��ǰ����1�����ܱ�����2������ǰ����3��,���ο���������Ϊȷ�����������ο���������Ϊ����ǰ��;
	//3.Init GMM: ���岢��ʼ��GMM(����ģ����ɷָ�Ҳ�ɵõ�����������GMM��ɻ�ӷ֣�
	//4.Sample Points:ǰ������ɫ���������о��ࣨ������kmeans���������෽��Ҳ��)
	//5.Learn GMM(���ݾ������������ÿ��GMM����еľ�ֵ��Э����Ȳ�����
	//4.Construct Graph������t-weight(�������n-weight��ƽ�����
	//7.Estimate Segmentation(����maxFlow����зָ�)
	//8.Save Result�������������mask�������mask��ǰ�������Ӧ�Ĳ�ɫͼ�񱣴����ʾ�ڽ��������У�
	
}