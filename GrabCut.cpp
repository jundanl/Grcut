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
//一.参数解释：
	//输入：
	 //cv::InputArray _img,     :输入的color图像(类型-cv:Mat)
     //cv::Rect rect            :在图像上画的矩形框（类型-cv:Rect) 
  	//int iterCount :           :每次分割的迭代次数（类型-int)


	//中间变量
	//cv::InputOutputArray _bgdModel ：   背景模型（推荐GMM)（类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）
	//cv::InputOutputArray _fgdModel :    前景模型（推荐GMM) （类型-13*n（组件个数）个double类型的自定义数据结构，可以为cv:Mat，或者Vector/List/数组等）


	//输出:
	//cv::InputOutputArray _mask  : 输出的分割结果 (类型： cv::Mat)

//二. 伪代码流程：
	//1.Load Input Image: 加载输入颜色图像;
	//2.Init Mask: 用矩形框初始化Mask的Label值（确定背景：0， 确定前景：1，可能背景：2，可能前景：3）,矩形框以外设置为确定背景，矩形框以内设置为可能前景;
	//3.Init GMM: 定义并初始化GMM(其他模型完成分割也可得到基本分数，GMM完成会加分）
	//4.Sample Points:前背景颜色采样并进行聚类（建议用kmeans，其他聚类方法也可)
	//5.Learn GMM(根据聚类的样本更新每个GMM组件中的均值、协方差等参数）
	//4.Construct Graph（计算t-weight(数据项）和n-weight（平滑项））
	//7.Estimate Segmentation(调用maxFlow库进行分割)
	//8.Save Result输入结果（将结果mask输出，将mask中前景区域对应的彩色图像保存和显示在交互界面中）
	
}