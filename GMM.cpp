#include "GMM.h"
#include "Kmeans.h"
//默认构造
GMM::GMM()
{
	data = new double[dataCount];
	//排列顺序： 1个权重，3个均值，9个协方差
	coefs = this->data;
	mean = coefs + componentsCount;
	cov = mean + channelsCount * componentsCount;
}

GMM::~GMM()
{
	delete[] data;
}

//使用外部数据构造
GMM::GMM(Mat& _model)
{
	checkModel(_model);

	data = new double[dataCount];
	//利用_model数据初始化
	double *base = _model.ptr<double>(0);
	memcpy(data, base, sizeof(double) * dataCount);

	//排列顺序： 1个权重，3个均值，9个协方差
	coefs = this->data;
	mean = coefs + componentsCount;
	cov = mean + channelsCount * componentsCount;
}

//从外部model拷贝数据
void GMM::copyFrom(Mat& _model)
{
	checkModel(_model);

	memcpy(data, _model.ptr<double>(0), sizeof(double) * dataCount);
}

//拷贝数据到外部model
void GMM::copyTo(Mat& _model)
{
	checkModel(_model);

	memcpy(_model.ptr<double>(0), data, sizeof(double) * dataCount);
}

//检测model有没有初始化,如果已经初始化检查类型,如果没有初始化进行初始化
void GMM::checkModel(Mat& _model)
{
	const int modelSize = 1 + channelsCount + channelsCount * channelsCount;
	if (_model.empty()) 
	{
		_model.create(1, modelSize * componentsCount, CV_64FC1);
		_model.setTo(Scalar(0));
	}
	else if (_model.type() != CV_64FC1 || _model.rows != 1 || _model.cols != modelSize * componentsCount)
	{
		CV_Error(CV_StsBadArg, "_model must be in 1 * 13 double type.");

	}
}

//更新GMM参数
void GMM::update() 
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
	}

}
//增加一个像素点
void GMM::add(int c, const Vec3d pixel)
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

//初始化，全部置零
void GMM::init() 
{
	memset(sum, 0, sizeof(double) * componentsCount * channelsCount);
	memset(product, 0, sizeof(double) * componentsCount * channelsCount * channelsCount);
	memset(counts, 0, sizeof(double) * componentsCount);
	samplesCount = 0;
}

void GMM::initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, Mat& bgdModel, Mat& fgdModel)
{
	/*----------------Kmeans 聚类像素-----------------*/
	const int kmeansItCount = 10; //kmeans最大迭代次数
	const int kmeasType = KMEANS_PP_CENTERS; //kmeans初始中心选取方法
	const float kmeansEpsilon = 0.0; //kmeans终止迭代的误差

	printf("%x %x\n", bgdGMM.data, bgdGMM.coefs);

	vector<Vec3f> bgdSamples, fgdSamples;
	Mat bgdLabels, fgdLabels;

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
	//kmeans(_bgdSamples, GMM::componentsCount, bgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);
	//kmeans(_fgdSamples, GMM::componentsCount, fgdLabels, TermCriteria(CV_TERMCRIT_ITER, kmeansItCount, kmeansEpsilon), 0, kmeasType);
	
	//使用自建kmeans进行聚类分析
	k_means(bgdSamples, bgdLabels, 10, 5, 3);
	k_means(fgdSamples, fgdLabels, 10, 5, 3);

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

	//拷贝数据到外部model
	bgdGMM.copyTo(bgdModel);
	fgdGMM.copyTo(fgdModel);
}