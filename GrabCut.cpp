#include "GrabCut.h"

class GMM
{
public:
	static const int componentsCount = 5; //5个高斯模型
	static const int channelsCount = 3; //处理3通道图片
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

		//排列顺序： 1个权重，3个均值，9个协方差
		coefs = model.ptr<double>(0);
		mean = coefs + componentsCount;
		cov = mean + channelsCount * componentsCount;
	}

	//初始化，全部置零
	void init() 
	{
		memset(sum, 0, sizeof(double) * componentsCount * channelsCount);
		memset(product, 0, sizeof(double) * componentsCount * channelsCount * channelsCount);
		memset(counts, 0, sizeof(double) * componentsCount);
		samplesCount = 0;
	}

	//增加一个像素点
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

	//更新GMM参数
	void update() 
	{
		for (int c = 0; c < componentsCount; c++)
		{
			if (counts[c] == 0) {
				coefs[c] = 0;
				continue;
			}
			//第c个高斯模型的贡献
			coefs[c] = (double)counts[c] / (double)samplesCount;

			//计算第c个高斯模型的期望
			int meanBase = c * channelsCount;
			for (int i = 0; i < channelsCount; i++)
			{
				mean[meanBase + i] = sum[c][i] / counts[c];
			}

			//计算第c个高斯模型的协方差
			double *co = cov + c * channelsCount * channelsCount;
			for (int i = 0; i < channelsCount; i++)
				for (int j = 0; j < channelsCount; j++)
				{
					co[i * channelsCount + j] =
						product[c][i][j] / counts[c] - mean[meanBase + i] * mean[meanBase + j];

				}
			
			//计算第c个高斯模型的协方差逆矩阵和行列式
			det[c] = co[0] * (co[4] * co[8] - co[5] * co[7])
				- co[1] * (co[3] * co[8] - co[5] * co[6])
				+ co[2] * (co[3] * co[7] - co[4] * co[6]);
			//为了计算逆矩阵，为行列式小于等于0的矩阵对角线增加噪声
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
			
			//计算逆
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
	/*----------------Kmeans 聚类像素-----------------*/
	const int kmeansItCount = 10; //kmeans最大迭代次数
	const int kmeasType = KMEANS_RANDOM_CENTERS; //kmeans初始中心选取方法
	const float kmeansEpsilon = 0.0; //kmeans终止迭代的误差

	Mat bgdLabels, fgdLabels;
	vector<Vec3f> bgdSamples, fgdSamples;

	//根据框选分配到两个GMM
	for (int i = 0; i < img.rows; i++)
		for (int j = 0; j < img.cols; j++)
		{
			if (mask.at<uchar>(i, j) == GC_BGD || mask.at<uchar>(i, j) == GC_PR_BGD)
				bgdSamples.push_back(img.at<Vec3b>(i, j));
			else
				fgdSamples.push_back(img.at<Vec3b>(i, j));
		}

	//进行Kmeans聚类分析
	CV_Assert(!bgdSamples.empty() && !fgdSamples.empty());
	Mat _bgdSamples(bgdSamples.size(), GMM::channelsCount, CV_32FC1, &bgdSamples[0][0]);
	Mat _fgdSamples(fgdSamples.size(), GMM::channelsCount, CV_32FC1, &fgdSamples[0][0]);
	kmeans(_bgdSamples, GMM::componentsCount, bgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);
	kmeans(_fgdSamples, GMM::componentsCount, fgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);

	/*-------------利用聚类结果更新GMM参数-------------*/
	//初始化
	bgdGMM.init();
	fgdGMM.init();
	
	//根据聚类标签为每个高斯模型分配采样点
	for (int i = 0; i < bgdSamples.size(); i++)
	{
		bgdGMM.add(bgdLabels.at<int>(i, 0), bgdSamples[i]);
	}
	for (int i = 0; i < fgdSamples.size(); i++)
	{
		fgdGMM.add(fgdLabels.at<int>(i, 0), fgdSamples[i]);
	}

	//计算参数
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

	/*-------------设置前后景mask---------------*/
	if (mode == GC_WITH_RECT)
	{
		cout << "GC_WITH_RECT" << endl;
		setRectInMask(img, mask, rect);
	}

	/*------------根据框选计算GMM-------------*/
	initGMM(img, bgdGMM, fgdGMM, mask);

	if ((mode != GC_CUT) || (iterCount == 0))
		return;


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