#include "GMM.h"
//Ĭ�Ϲ���
GMM::GMM()
{
	//����˳�� 1��Ȩ�أ�3����ֵ��9��Э����
	coefs = data;
	mean = data + componentsCount;
	cov = mean + channelsCount * componentsCount;
}

//ʹ���ⲿ���ݹ���
GMM::GMM(Mat& _model)
{
	checkModel(_model);

	//����_model���ݳ�ʼ��
	double *base = _model.ptr<double>(0);
	memcpy(data, base, sizeof(double) * dataCount);

	//����˳�� 1��Ȩ�أ�3����ֵ��9��Э����
	coefs = data;
	mean = data + componentsCount;
	cov = mean + channelsCount * componentsCount;
}

//���ⲿmodel��������
void GMM::copyFrom(Mat& _model)
{
	checkModel(_model);

	memcpy(data, _model.ptr<double>(0), sizeof(double) * dataCount);
}

//�������ݵ��ⲿmodel
void GMM::copyTo(Mat& _model)
{
	checkModel(_model);

	memcpy(_model.ptr<double>(0), data, sizeof(double) * dataCount);
}

//���model��û�г�ʼ��,����Ѿ���ʼ���������,���û�г�ʼ�����г�ʼ��
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

//����GMM����
void GMM::update() 
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
//����һ�����ص�
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

//��ʼ����ȫ������
void GMM::init() 
{
	memset(sum, 0, sizeof(double) * componentsCount * channelsCount);
	memset(product, 0, sizeof(double) * componentsCount * channelsCount * channelsCount);
	memset(counts, 0, sizeof(double) * componentsCount);
	samplesCount = 0;
}

void GMM::initGMM(const Mat& img, const Mat& mask, GMM& bgdGMM, GMM& fgdGMM, Mat& bgdModel, Mat& fgdModel)
{
	/*----------------Kmeans ��������-----------------*/
	const int kmeansItCount = 10; //kmeans����������
	const int kmeasType = KMEANS_PP_CENTERS; //kmeans��ʼ����ѡȡ����
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

	//�������ݵ��ⲿmodel
	bgdGMM.copyTo(bgdModel);
	fgdGMM.copyTo(fgdModel);
}