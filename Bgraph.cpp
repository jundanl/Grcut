#include "Bgraph.h"
void buildgraph::mincut(Mat &mask, const Mat &img, const Mat &bg, const Mat &fg)
{
	if (img.type() == CV_8UC3)
	{
		int rows = img.rows;
		int cols = img.cols;
		initparam(bg, fg, img);//初始化各项参数
		//把各类边的权重存在一个和img大小一样的矩阵里
		rweight.create(Size(cols, rows), CV_64FC1);
		rweight.setTo(Scalar(0));
		cweight.create(Size(cols, rows), CV_64FC1);
		cweight.setTo(Scalar(0));
		downweight.create(Size(cols, rows), CV_64FC1);
		downweight.setTo(Scalar(0));
		upweight.create(Size(cols, rows), CV_64FC1);
		upweight.setTo(Scalar(0));
		//计算能量函数第二项
		calbeta(img);
		calborderweight(img);
		
		formatgraph(mask, img);//建立图
		//cout << "flag1" << endl;
		
		g->maxflow();//对图进行mincut分割
		//cout << "flag2" << endl;
		for (int i = 0; i < rows; i++)
			for (int j = 0; j < cols; j++)
			{
				if (mask.at<uchar>(i,j) == GC_PR_BGD || mask.at<uchar>(i,j) == GC_PR_FGD)
				{
					if (g->what_segment(i*cols + j) == GraphType::SOURCE)
						mask.at<uchar>(i, j) = GC_FGD;
					else
						mask.at<uchar>(i, j) = GC_BGD;
				}
			}
		//cout << "flag3" << endl;
	}
	else
		cout << "！错误：图片的类型必须为CV_8UC3" << endl;

}
void buildgraph::initparam(const Mat &bg, const Mat &fg,const Mat &img)
{
	points = img.rows*img.cols;
	edge = 4 * img.rows*img.cols - 3 * img.rows - 3 * img.cols + 2;
	backg = bg;
	foreg = fg;
	//te.create(1, 3, CV_64FC1);
	//inv.create(3, 3, CV_64FC1);
}

void buildgraph::calbeta(const Mat &img)
{
	double sum = 0;
	int x = img.cols;
	int y = img.rows;
	int count = 0;
	
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
		{
			Vec3d point = img.at<Vec3b>(i, j);
			//每一个点计算右上，右，右下，下四条边，以保证边不重复计算
			if (i < y - 1)//下
			{
				Vec3d d = img.at<Vec3b>(i + 1, j);
				sum += distance(point, d);
				count++;
			}
			if (j < x - 1)//右
			{
				Vec3d d = img.at<Vec3b>(i, j + 1);
				sum += distance(point, d); count++;
			}
			if (j<x-1&&i<y-1)//右下
			{
				Vec3d d = img.at<Vec3b>(i+1, j + 1);
				sum += distance(point, d); count++;
			}
			if (j < x - 1&&i>0)//右上
			{
				Vec3d d = img.at<Vec3b>(i-1, j + 1);
				sum += distance(point, d); count++;
			}
		}
	assert(edge > 0);
	if (sum < numeric_limits<double>::epsilon())
		beta = 0;
	else
		beta = 1.0 / (2 * sum / edge);
	//std::cout << "edge:" <<edge<< endl;
	//std::cout << "beta:" << beta << endl;
	//std::cout << "count:" << count << endl;
}
double buildgraph::distance(Vec3d p,Vec3d d)
{
	//static int count = 0;
	Vec3d point = p - d;
	double sum = 0;
	for (int i = 0; i < 3; i++)
		sum += point[i] * point[i];
	//if (count >= 2000&&count<=30000){
		//Vec3b 
	//	cout << "dis1:" << sum;
	//	cout << "dis2:" << point.dot(point) << endl;;
		
	//}
	//count++;
	return sum;
}
void buildgraph::calborderweight(const Mat &img)
{
	double dis = sqrt(1.0 * 2);
	int x = img.cols;
	int y = img.rows;
	for (int i = 0; i < y; i++)
		for (int j = 0; j < x; j++)
		{
			Vec3d point = img.at<Vec3b>(i, j);
			//每一个点计算右上，右，右下，下四条边的权重，并按类别保存在矩阵中
			if (i < y - 1)//下，保存在cweight中
			{
				Vec3d d = img.at<Vec3b>(i + 1, j);
				cweight.at<double>(i,j)= gamma*exp(-1.0*beta*distance(point, d));
			}
			if (j < x - 1)//右，保存在rweight中
			{
				Vec3d d = img.at<Vec3b>(i, j + 1);
				rweight.at<double>(i, j) = gamma*exp(-1.0*beta*distance(point, d));
			}
			if (j<x - 1 && i<y - 1)//右下，保存在downweight中
			{
				Vec3d d = img.at<Vec3b>(i + 1, j + 1);
				downweight.at<double>(i, j) = gamma*exp(-1.0*beta*distance(point, d))/dis;
			}
			if (j < x - 1 && i>0)//右上，保存在upweight中
			{
				Vec3d d = img.at<Vec3b>(i - 1, j + 1);
				upweight.at<double>(i, j) = gamma*exp(-1.0*beta*distance(point, d))/dis;
			}
		}
}
double buildgraph::calplainweight(const Vec3b &point, int label)
{
	double sum = 0;
	for (int i = 0; i < components; i++)
	{
		sum += calsingleweight(label ? foreg : backg, i,point);
	}
	return -log(sum);
	//return sum;
}
double buildgraph::calsingleweight(const Mat &model, int label,const Vec3b &point)
{
	//static int bb = 5;
	double result=0;
	//int size = 13;
	double coefs=model.at<double>(0,label);//每个高斯模型的贡献
	//test = coefs;
	
	if (coefs <= 0)
		return result;

	Mat mean(1,3,CV_64FC1);//每个高斯模型的期望与点的差
	Mat cov(3,3,CV_64FC1);//每个高斯模型的方差
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			cov.at<double>(i, j) = model.at<double>(0, label*9+3 * i + j+5+15);
		mean.at<double>(0, i) = model.at<double>(0, 3*label + 5 + i)-(double)point[i];
		//te.at<double>(0, i) = model.at<double>(0, 3 * label + 5 + i);
	}
	/*if (bb++ < 5){
		//cout << "model" << model << endl;
		//cout << "label:" << label << endl;
		cout << "mean:" << mean << endl;
		cout << "cov:" << cov << endl;
		cout << "coefs:" << coefs << endl;
	}*/
	//result += -log(coefs);
	//result += log(determinant(cov)) / 2;//行列式的值
	//inv = cov.inv();
	
	Mat temp=mean*(cov.inv())*(mean.t());
	//det = temp.at<double>(0, 0);
	//result += temp.at<double>(0, 0) / 2;
	result = coefs*1.0f / sqrt(determinant(cov))*exp(-0.5*temp.at<double>(0,0));


	return result;
}

void buildgraph::formatgraph(const Mat &mask, const Mat &img)
{
	//static int aa = 0;
	int rows = img.rows;
	int cols = img.cols;
	g = new GraphType(points, edge);
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
		{
			int index = g->add_node();
			double source;
			double sink;
			Vec3b pixel = img.at<Vec3b>(i, j);
			int label = mask.at<uchar>(i, j);
			if (label == GC_PR_BGD || label == GC_PR_FGD)//点为可能前景或可能后景
			{
				source = calplainweight(pixel, GC_BGD);
				sink = calplainweight(pixel, GC_FGD);
				
			}
			else if (label == GC_BGD)//点为后景
			{
				source = 0;
				sink = lama;
			}
			else//点为前景
			{
				source = lama;
				sink = 0;
			}
			/*if (aa++ > 25000 && aa < 25003){
			cout << "S:" << source << " " << "t:" << sink << endl;
			cout << "S:" << i << " " << "t:" << j << endl;
			cout << "coefs:" << test << endl;
			cout << "mean " << det << endl;
		}*/
			g->add_tweights(index, source, sink);//添加与源点和汇点的边
			//添加与领域像素点的边
			if (i > 0 && j < cols - 1)//右上的边
			{
				double weight = upweight.at<double>(i, j);
				g->add_edge(index, index - cols + 1,weight, weight);
			}
			if (i>0 && j > 0)//左上的边
			{
				double weight = downweight.at<double>(i - 1, j - 1);
				g->add_edge(index, index - cols - 1, weight, weight);
			}
			if (i>0)//上边
			{
				double weight = cweight.at<double>(i - 1, j);
				g->add_edge(index, index - cols, weight, weight);
			}
			if (j > 0)//左边
			{
				double weight = rweight.at<double>(i, j - 1);
				g->add_edge(index, index - 1, weight, weight);
			}
		}
}