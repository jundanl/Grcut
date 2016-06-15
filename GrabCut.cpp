#include "GrabCut.h"

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