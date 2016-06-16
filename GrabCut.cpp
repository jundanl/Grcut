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

//void initGMM(const Mat& img, const Mat& mask, vector<Vec3b>& bgdSamples, vector<Vec3b>& fgdSamples )
//{
//	for (int i = 0; i < mask.rows; i++)
//		for (int j = 0; j < mask.cols; j++)
//		{
//			if (mask.at<uchar>(i, j) & 1 == 1)
//				fgdSamples.push_back(img.at<Vec3b>(i, j));
//			else
//				bgdSamples.push_back(img.at<Vec3b>(i, j));
//		}
//}

void GrabCut2D::GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect, cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel, int iterCount, int mode )
{
    std::cout<<"Execute GrabCut Function: Please finish the code here!"<<std::endl;

	const Mat& img = _img.getMat();
	Mat& mask = _mask.getMatRef();
	Mat& bgdModel = _bgdModel.getMatRef();
	Mat& fgdModel = _fgdModel.getMatRef();
	/*iterCount = 2;
	grabCut(img,mask,rect,bgdModel,fgdModel,iterCount,mode);
	return;
*/
	static GMM bgdGMM = GMM(bgdModel);
	static GMM fgdGMM = GMM(fgdModel);
	if (mode != GC_CUT)
	{
		if (mode == GC_WITH_RECT)
		{
			cout << "GC_WITH_RECT" << endl;
			setRectInMask(img, mask, rect);
		}
		//bgdGMM = GMM();
		//fgdGMM = GMM();
		GMM::initGMM(img, mask, bgdGMM, fgdGMM, bgdModel, fgdModel);
	}
	if ((mode != GC_CUT) || (iterCount == 0))
		return;
	cout << "befot initGMM" << endl;
	GMM::initGMM(img, mask, bgdGMM, fgdGMM, bgdModel, fgdModel);
	//GMM().initGMM(img,mask,bgdGMM,fgdGMM,bgdModel,fgdModel);
	cout << "after initGMM" << endl;
	//return;
	buildgraph bg;
	bg.mincut(mask,img,bgdModel,fgdModel);
	cout << "after mincut" << endl;
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