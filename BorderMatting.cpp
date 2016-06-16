#include "BorderMatting.h"


BorderMatting::BorderMatting()
{
}


BorderMatting::~BorderMatting()
{
}

bool outrange(int x, int l, int r)
{
	if (x<l || x>r)
		return true;
	else
		return false;
}

void BorderDetection(const Mat& img, Mat& rs)
{
	Mat edge;
	Canny(img, edge, 3, 9, 3);
	edge.convertTo(rs, CV_8UC1);
}


void BorderMatting::dfs(int x, int y, const Mat& edge, Mat& color, Contour& contour)
{
	para_point pt;
	pt.p.x = x; pt.p.y = y;
	pt.index = areacnt++;
	pt.section = sections;
	contour.push_back(pt);
	for (int i = 0; i < nstep; i++)
	{
		int zx = nx[i];
		int zy = ny[i];
		int newx = x + zx;
		int newy = y + zy;
		if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))
			continue;
		if (edge.at<uchar>(newx,newy) == 0 || color.at<uchar>(newx,newy) != 0)
			continue;
		color.at<uchar>(newx, newy) = 255;
		dfs(newx,newy,edge,color,contour);
	}
}

void BorderMatting::ParameterizationContour(const Mat& edge,Contour& contour)
{
	int rows = edge.rows;
	int cols = edge.cols;

	Mat color(edge.size(), edge.type(), Scalar(0));
	sections = 0;
	areacnt = 0;
	tot = 0;
	bool flag = false;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (edge.at<uchar>(i, j) != 0)
			{
				if (color.at<uchar>(i, j) != 0)
					continue;
				if (flag)
					sections++;
				dfs(i, j, edge, color,contour);
				flag = true;
			}	
}

void BorderMatting::StripInit(const Mat& mask, Contour& contour, Strip& strip)
{
	Mat color(mask.size(), CV_32SC1, Scalar(0));
	vector<point> queue;
	for (int i = 0; i < contour.size(); i++)
	{
		inf_point ip;
		ip.p = contour[i].p;
		ip.dis = 0;
		ip.area = contour[i].index;
		queue.push_back(ip.p);
		strip[ip.p.x*COE + ip.p.y] = ip;
		color.at<int>(ip.p.x, ip.p.y) = ip.area+1;
	}
	int l = 0;
	while (l < queue.size())
	{
		point p = queue[l++];
		inf_point ip = strip[p.x*COE+p.y];
		if (abs(ip.dis) >= 6)
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))
				continue;
			inf_point nip;
			if (color.at<int>(newx, newy) != 0){
				if (ip.area % 6 != 0)
					continue;
				if ((color.at<int>(newx, newy) - 1) % 6 == 0)
					continue;
				nip = strip[newx*COE+newy];
			}
			else
			{
				nip.p.x = newx; nip.p.y = newy;
			}
			nip.dis = abs(ip.dis) + 1;
			if ((mask.at<uchar>(newx, newy) & 1) != 1 )
			{
				nip.dis = -nip.dis;
			}
			nip.area = ip.area;
			strip[nip.p.x*COE + nip.p.y] = nip;
			queue.push_back(nip.p);
			color.at<int>(newx, newy) = nip.area+1;
		}
	}
}


double Gaussian(double x, double delta, double sigma)
{
	//const double PI = 3.14159;
	double rs = -(x - delta) / sigma;
	rs = exp(rs);
	rs = 1.0 / (1.0+rs);
	return rs;
	//double e = exp(-(pow(x-delta,2.0)/(2.0*sigma)));
	//double rs = 1.0 / (pow(sigma,0.5)*pow(2.0*PI, 0.5))*e;
	//return rs;
}

double ufunc(double a,double uf,double ub)
{
	return (1 - a)*uf + a*ub;
}

double cfunc(double a, double cf,double cb)
{
	return pow(1 - a, 2.0)*cf + pow(a, 2.0)*cb;
}

double BorderMatting:: Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, Strip& strip, const Mat& gray)
{
	vector<inf_point> queue;
	map<int, bool> color;
	double sum = 0;
	inf_point ip = strip[p.x*COE + p.y];
	double alpha = Gaussian(ip.dis / 6.0, delta, sigma);
	double D = Gaussian(gray.at<float>(ip.p.x, ip.p.y), ufunc(alpha, uf, ub), cfunc(alpha, cf, cb));
	D = -log(D) / log(2.0);
	sum += D;
	queue.push_back(ip);
	color[p.x*COE + p.y] = true;
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++];
		if (abs(ip.dis) >= 6)
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))
				continue;
			if (color[newx*COE+newy])
				continue;
			inf_point newip = strip[newx*COE+newy];
			if (newip.area != index)
				continue;
			double alpha = Gaussian(newip.dis / 6.0,delta,sigma);
			double D = Gaussian(gray.at<float>(newx,newy),ufunc(alpha,uf,ub),cfunc(alpha,cf,cb));
			D = - log(D) / log(2.0);
			sum += D;
			queue.push_back(newip);
			color[newx*COE + newy] = true;
		}
	}
	return sum;
	//return 1;
}

void calculate(point p,const Mat& gray, const Mat& mask,double& uf,double& ub, double& cf, double& cb)
{
	const int len = 20;
	double sumf=0, sumb=0;
	int cntf = 0, cntb = 0;
	int rows = gray.rows;
	int cols = gray.cols;
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if  (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
				{
					int g = gray.at<float>(x, y);
					if ((mask.at<uchar>(x, y) & 1) == 0)
					{
						sumb += g;
						cntb++;
					}
					else
					{
						sumf += g;
						cntf++;
					}
				}
	uf = (double)sumf / (double)cntf;
	ub = (double)sumb / (double)cntb;
	cf = 0;
	cb = 0;
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
			{
				int g = gray.at<float>(x, y);
				if ((mask.at<uchar>(x, y) & 1) == 0)
				{
					cb += pow(g - ub, 2.0);
				}
				else
				{
					cf += pow(g - uf, 2.0);
				}
			}
	cf /= cntf;
	cb /= cntb;
}

void BorderMatting::EnergyMinimization(const Mat& oriImg, const Mat& mask, Contour& contour,Strip& strip)
{
	Mat gray;
	cvtColor(oriImg, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray,CV_32FC1,1.0/255.0);
	int index;
	for (int i = 0; i < contour.size(); i++)
	{
		para_point pp = contour[i];
		index = pp.index;
		double uf,ub, cf,cb;
		calculate(pp.p,gray,mask,uf,ub,cf,cb);
		for (int d0 = 0; d0< deltaLevels; d0++)
			for (int s0 = 0; s0 < sigmaLevels; s0++)
			{
				double sigma0 = sigma*(s0 + 1);
				double delta0 = delta*(d0 + 1);
				ef[index][d0][s0] = MAXNUM;
				double D = Dfunc(index, pp.p, uf, ub, cf, cb, delta0, sigma0, strip, gray);
				if (index == 0)
				{
					ef[index][d0][s0] = D;
					continue;
				}
				if (index % 6 == 0)
				{
					double rs = ef[index - 1][d0][s0] +  D;
					ef[index][d0][s0] = rs;
					dands ds;
					ds.delta = d0;
					ds.sigma = s0;
					rec[index][d0][s0] = ds;
					continue;
				}
				for (int d1 = 0; d1 < deltaLevels; d1++)
					for (int s1 = 0; s1 < sigmaLevels; s1++)
					{
						double delta1 = delta*(d1 + 1);
						double sigma1 = sigma*(s1 + 1);
						double rs = ef[index-1][d1][s1] + Vfunc(delta0 - delta1, sigma0 - sigma1) + D;
						if (rs < ef[index][d0][s0])
						{
							dands ds;
							ds.sigma = s1; ds.delta = d1;
							ef[index][d0][s0] = rs;
							rec[index][d0][s0] = ds;
						}
					}
			}
	}
	double minE = MAXNUM;
	dands ds;
	vecds = vector<dands>(index);
	for (int d0 = 0; d0< deltaLevels; d0++)
		for (int s0 = 0; s0 < sigmaLevels; s0++)
		{
			if (ef[index][d0][s0] < minE)
			{
				minE = ef[index][d0][s0];
				ds.delta = d0;
				ds.sigma = s0;
			}
		}
	vecds[index]=ds;
	for (int i = index - 1; i >= 0; i--)
	{
		dands ds0 = vecds[vecds.size()-1];
		dands ds = rec[i + 1][ds0.delta][ds0.sigma];
		vecds[i]=ds;
	}
}

void BorderMatting::CalculateMask(Mat& bordermask, const Mat& mask)
{
	bordermask = Mat(mask.size(), CV_32FC1, Scalar(0));
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if ((mask.at<uchar>(i, j) & 1) == 1)
				bordermask.at<float>(i, j) = 1;
	for (int i = 0; i < strip.size(); i++)
	{
		inf_point ip= strip[i];
		int index = ip.area;
		dands ds = vecds[index];
		double dt = (ds.delta + 1)*delta;
		double sg = (ds.sigma + 1)*sigma;
		double alpha = Gaussian(ip.dis / 6.0, 0.5, 1);;
		bordermask.at<float>(ip.p.x, ip.p.y) = alpha;
	}
}

void BorderMatting::borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask)
{
	init(oriImg);
	Mat edge;
	edge = mask & 1;
	edge.convertTo(edge, CV_8UC1, 255);
	BorderDetection(edge,edge);
	Mat amask(edge.size(),edge.type(),Scalar(0));
	ParameterizationContour(edge,contour);
	Mat test;
	mask.convertTo(test,CV_8UC1);
	StripInit(test,contour,strip);
	//DPPreProcess();
	EnergyMinimization(oriImg,mask,contour,strip);
	CalculateMask(borderMask,mask);
	
	//for (int i = 0; i < rows; i++)
	//	for (int j = 0; j < cols; j++)
	//		if (borderMask.at<float>(i, j) != 0)
	//			cout << borderMask.at<float>(i, j) << endl;
	GaussianBlur(borderMask, borderMask, Size(7, 7), 5.0);
	vector<Mat> ch_img(3);
	vector<Mat> ch_bg(3);
	Mat img;
	oriImg.convertTo(img,CV_32FC3,1.0/255.0);
	cv::split(img, ch_img);
	Mat bg = Mat(img.size(), CV_32FC3 ,Scalar(1.0,1.0,1.0));
	cv::split(bg, ch_bg);
	ch_img[0] = ch_img[0].mul(borderMask) + ch_bg[0].mul(1.0 - borderMask);
	ch_img[1] = ch_img[1].mul(borderMask) + ch_bg[1].mul(1.0 - borderMask);
	ch_img[2] = ch_img[2].mul(borderMask) + ch_bg[2].mul(1.0 - borderMask);
	Mat res;
	cv::merge(ch_img, res);
	imshow("result", res);
	//imshow("result", res);
}

void BorderMatting::init(const Mat& img)
{
	rows = img.rows;
	cols = img.cols;
}
