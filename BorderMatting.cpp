#include "BorderMatting.h"


BorderMatting::BorderMatting()
{
}

BorderMatting::~BorderMatting()
{
}

inline bool outrange(int x, int l, int r)
{
	if (x<l || x>r)
		return true;
	else
		return false;
}

/*������ʼ��*/
void BorderMatting::init(const Mat& img)
{
	rows = img.rows;
	cols = img.cols;
	sections = 0;
	areacnt = 0;
	tot = 0;
	contour.clear();
	strip.clear();
	vecds.clear();
}

/*��Ե���*/
void BorderDetection(const Mat& img, Mat& rs)
{
	Mat edge;
	Canny(img, edge, 3, 9, 3);
	edge.convertTo(rs, CV_8UC1);
}

/*���ѱ�������*/
void BorderMatting::dfs(int x, int y, const Mat& edge, Mat& color)
{
	color.at<uchar>(x, y) = 255;//����ѱ���
	para_point pt;
	pt.p.x = x; pt.p.y = y; //����
	pt.index = areacnt++;//��������ÿһ����������index
	pt.section = sections;//��������
	contour.push_back(pt); //��������vector
	for (int i = 0; i < nstep; i++)//ö��(x,y)���ڵ�
	{
		int zx = nx[i];
		int zy = ny[i];
		int newx = x + zx;
		int newy = y + zy;
		if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) //����ͼ��Χ
			continue;
		if (edge.at<uchar>(newx,newy) == 0 || color.at<uchar>(newx,newy) != 0) //���������ϵĵ㣬�����Ѿ���������
			continue;
		dfs(newx,newy,edge,color);//��(newx,newy)�������������ѱ�������
	}
}

/*����������*/
void BorderMatting::ParameterizationContour(const Mat& edge)
{
	int rows = edge.rows;
	int cols = edge.cols;
	sections = 0; //���������ĸ���
	areacnt = 0; //�������ϲ�ͬ��Ϊ���ĵ�����������������ϵ�ĸ�����
	Mat color(edge.size(), CV_8UC1, Scalar(0));//�������
	bool flag = false;
	for (int i = 0; i < rows; i++)
		for (int j = 0; j < cols; j++)
			if (edge.at<uchar>(i, j) != 0)//(i,j)�������ϵĵ�
			{
				if (color.at<uchar>(i, j) != 0)//�������Ѿ���������
					continue;
				/*(i,j)���������߻�û�б�������*/
				dfs(i, j, edge, color);//���ѱ�������
				sections++;//����������1
			}	
}

/*��������*/
void BorderMatting::StripInit(const Mat& mask)
{
	Mat color(mask.size(), CV_32SC1, Scalar(0));//�������

	/*���������������ѱ����������������������򡪡�������Ӧ������������*/
	//��ʼ�����У��������������е�
	vector<point> queue;
	for (int i = 0; i < contour.size(); i++)
	{
		inf_point ip;
		ip.p = contour[i].p; //����
		ip.dis = 0; //�������ĵ��ŷ�Ͼ���
		ip.area = contour[i].index; //��������
		strip[ip.p.x*COE + ip.p.y] = ip; //�������������key��hash��ֵΪ������
		queue.push_back(ip.p); //����������
		color.at<int>(ip.p.x, ip.p.y) = ip.area+1; //������ǣ������+1
	}
	//���ѱ�������
	int l = 0;
	while (l < queue.size())
	{
		point p = queue[l++]; //ȡ����
		inf_point ip = strip[p.x*COE+p.y]; //��strip�еõ������Ϣ
		if (abs(ip.dis) >= stripwidth) //ֻ���������ڵĵ�
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//ö�����ڵ�
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))//����ͼ��Χ
				continue;
			inf_point nip;
			if (color.at<int>(newx, newy) != 0){//�õ�������������
				///*������ÿ6����ȡһ���ؼ��㡣���Ƚ��������������Щ�ؼ���Ϊ���ĵ�����*/
				//if (ip.area % 6 != 0) //��ǰ���ĵ㲻�ǹؼ���
				//	continue;
				//if ((color.at<int>(newx, newy) - 1) % 6 == 0) //��ǰ�������ڹؼ�������
				//	continue;
				//nip = strip[newx*COE+newy];
				continue;
			}
			else
			{
				nip.p.x = newx; nip.p.y = newy;
			}
			nip.dis = abs(ip.dis) + 1;//ŷʽ����+1
			if ((mask.at<uchar>(newx, newy) & 1) != 1 ) //���ڱ���
			{
				nip.dis = -nip.dis;
			}
			nip.area = ip.area;
			strip[nip.p.x*COE + nip.p.y] = nip; //��������
			queue.push_back(nip.p); //�������
			color.at<int>(newx, newy) = nip.area+1; //������ǣ������+1
		}
	}
}

/*��˹�ܶȺ���*/
inline double Gaussian(double x, double delta, double sigma)
{
	const double PI = 3.14159;
	double e = exp(-(pow(x-delta,2.0)/(2.0*sigma)));
	double rs = 1.0 / (pow(sigma,0.5)*pow(2.0*PI, 0.5))*e;
	return rs;
}

inline double ufunc(double a,double uf,double ub)
{
	return (1.0 - a)*ub + a*uf;
}

inline double cfunc(double a, double cf,double cb)
{
	return pow(1.0 - a, 2.0)*cb + pow(a, 2.0)*cf;
}

/*sigmoid����*/
inline double Sigmoid(double r, double delta, double sigma)
{
	double rs = -(r - delta) / sigma;
	rs = exp(rs);
	rs = 1.0 / (1.0 + rs);
	return rs;
}

inline double Dterm(inf_point ip, float I, double delta, double sigma, double uf, double ub, double cf, double cb )
{
	double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta, sigma);
	double D = Gaussian(I, ufunc(alpha, uf, ub), cfunc(alpha, cf, cb));
	D = -log(D) / log(2.0);
	return D;
}

/*����term D*/
double BorderMatting:: Dfunc(int index, point p, double uf, double ub, double cf, double cb, double delta, double sigma, const Mat& gray)
{
	vector<inf_point> queue;
	map<int, bool> color;
	double sum = 0;
	inf_point ip = strip[p.x*COE + p.y]; //��strip�л�ȡ���ĵ���Ϣ
	sum += Dterm(ip, gray.at<float>(ip.p.x, ip.p.y),delta,sigma,uf,ub,cf,cb);
	queue.push_back(ip);//�������
	color[ip.p.x*COE + ip.p.y] = true;//��Ǳ���
	/*���ѱ�����pΪ���ĵ������*/
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++];
		if (abs(ip.dis) >= stripwidth)
			break;
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//ö�����ڵ�
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1)) //����ͼ��Χ
				continue;
			if (color[newx*COE+newy])//�Ѿ�������
				continue;
			inf_point newip = strip[newx*COE+newy];//��strip�л�ȡ�����Ϣ
			if (newip.area == index) //������pΪ���ĵ������
			{
				sum += Dterm(newip, gray.at<float>(newx, newy), delta, sigma, uf, ub, cf, cb);
			}
			queue.push_back(newip);//�������
			color[newx*COE + newy] = true;//��Ǳ���
		}
	}
	return sum;
}

/*����L*L�����ǰ������ֵ�ͷ���*/
void calSampleMeanCovariance(point p, const Mat& gray, const Mat& mask, double& uf, double& ub, double& cf, double& cb)
{
	int len = L;
	double sumf=0, sumb=0;
	int cntf = 0, cntb = 0;
	int rows = gray.rows;
	int cols = gray.cols;
	//�����ֵ
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if  (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
				{
					float g = gray.at<float>(x, y);
					if ((mask.at<uchar>(x, y) & 1) == 0) //����
					{
						sumb += g;
						cntb++;
					}
					else //ǰ��
					{
						sumf += g;
						cntf++;
					}
				}
	uf = (double)sumf / (double)cntf; //ǰ����ֵ
	ub = (double)sumb / (double)cntb; //������ֵ
	//���㷽��
	cf = 0;
	cb = 0;
	for (int x = p.x - len; x <= p.x + len; x++)
		for (int y = p.y - len; y <= p.y + len; y++)
			if (!(outrange(x, 0, rows - 1) || outrange(y, 0, cols - 1)))
			{
				float g = gray.at<float>(x, y);
				if ((mask.at<uchar>(x, y) & 1) == 0) //����
				{
					cb += pow(g - ub, 2.0);
				}
				else //ǰ��
				{
					cf += pow(g - uf, 2.0);
				}
			}
	cf /= (double)cntf; //ǰ������
	cb /= (double)cntb; //��������
}

/*ͨ��level����sigma*/
inline double sigma(int level)
{
	return 0.025*(level);
}

/*ͨ��level����delta*/
inline double delta(int level)
{
	return 0.1*level;
	//return -1.0 + 0.2*level;
}

/*������С������̬�滮��ÿ�������delta��sigma*/
void BorderMatting::EnergyMinimization(const Mat& oriImg, const Mat& mask)
{
	//ת��Ϊ�Ҷ�ͼ
	Mat gray;
	cvtColor(oriImg, gray, COLOR_BGR2GRAY);
	gray.convertTo(gray,CV_32FC1,1.0/255.0);
	//������С����ÿ�������delta��sigma
	for (int i = 0; i < contour.size(); i++)//ö��������ÿһ���㣬���������������ĵ�
	{
		para_point pp = contour[i];
		int index = pp.index;
		double uf,ub,cf,cb;
		//��L*L�����ǰ������ֵ�ͷ���
		calSampleMeanCovariance(pp.p,gray,mask,uf,ub,cf,cb);
		for (int d0 = 0; d0< deltaLevels; d0++) //ö��delta
			for (int s0 = 0; s0 < sigmaLevels; s0++) //ö��sigma
			{
				double sigma0 = sigma(s0);
				double delta0 = delta(d0);
				ef[index][d0][s0] = MAXNUM;
				//����term D
				double D = Dfunc(index, pp.p, uf, ub, cf, cb, delta0, sigma0, gray);
				//������������:termD + termV
				if (index == 0)
				{
					ef[index][d0][s0] = D;
					continue;
				}
				//if (index % 6 != 0)//Ϊ�˼ӿ���㣬�ǹؼ��㲻��ö�١�d0,s0ȡ��(index-1)ʱ��ͬ
				//{
				//	ef[index][d0][s0] = ef[index - 1][d0][s0] + D;
				//	dands ds;
				//	ds.delta = d0;
				//	ds.sigma = s0;
				//	rec[index][d0][s0] = ds;
				//	continue;
				//}
				for (int d1 = 0; d1 < deltaLevels; d1++)//ö��index-1ʱ��delta
					for (int s1 = 0; s1 < sigmaLevels; s1++)//ö��index-1ʱ��sigma
					{
						double delta1 = delta(d1);
						double sigma1 = sigma(s1);
						double Vterm = 0;
						if (contour[i - 1].section == pp.section)//����һ������ͬһ����
						{
							Vterm = Vfunc(delta0 - delta1, sigma0 - sigma1);
						}
						double rs = ef[index-1][d1][s1] + Vterm + D;
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
	//����������Сֵ
	double minE = MAXNUM;
	dands ds;
	vecds = vector<dands>(areacnt);//��¼ÿ�������delta��sigma
	for (int d0 = 0; d0< deltaLevels; d0++)
		for (int s0 = 0; s0 < sigmaLevels; s0++)
		{
			if (ef[areacnt-1][d0][s0] < minE)
			{
				minE = ef[areacnt-1][d0][s0];
				ds.delta = d0;
				ds.sigma = s0;
			}
		}
	//��¼��������Сʱ��ÿ�������delta��sigma
	vecds[areacnt-1]=ds;
	for (int i = areacnt - 2; i >= 0; i--)
	{
		dands ds0 = vecds[i + 1];
		dands ds = rec[i + 1][ds0.delta][ds0.sigma];
		vecds[i]=ds;
	}
}

/*����alpha*/
inline double adjustA(double a)
{
	if (a < 0.01)
		return 0;
	if (a > 9.99)
		return 1;
	return a;
}

/*����ÿ�����ص��alpha*/
void BorderMatting::CalculateMask(Mat& bordermask, const Mat& mask)
{
	bordermask = Mat(mask.size(), CV_32FC1, Scalar(0));

	Mat color(mask.size(), CV_32SC1, Scalar(0));//�������

	/*���������������ѱ���ͼ�񣬼���alpha*/
	//��ʼ�����У��������������е�
	vector<inf_point> queue;
	for (int i = 0; i < contour.size(); i++)
	{
		inf_point ip;
		ip.p = contour[i].p; //����
		ip.dis = 0; //�������ĵ��ŷ�Ͼ���
		ip.area = contour[i].index; //��������
		queue.push_back(ip); //����������
		color.at<int>(ip.p.x, ip.p.y) = 1; //�������
		//����alpha
		dands ds = vecds[ip.area];
		double alpha = Sigmoid((double)ip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
		alpha = adjustA(alpha);//����alpha
		bordermask.at<float>(ip.p.x, ip.p.y) = (float)alpha;
	}
	//���ѱ�������
	int l = 0;
	while (l < queue.size())
	{
		inf_point ip = queue[l++]; //ȡ����
		int x = ip.p.x;
		int y = ip.p.y;
		for (int i = 0; i < rstep; i++)//ö�����ڵ�
		{
			int newx = x + rx[i];
			int newy = y + ry[i];
			if (outrange(newx, 0, rows - 1) || outrange(newy, 0, cols - 1))//����ͼ��Χ
				continue;
			inf_point nip;
			if (color.at<int>(newx, newy) != 0)
				continue;
			nip.p.x = newx; nip.p.y = newy;
			nip.dis = abs(ip.dis) + 1;//ŷʽ����+1
			if ((mask.at<uchar>(newx, newy) & 1) != 1) //���ڱ���
			{
				nip.dis = -nip.dis;
			}
			nip.area = ip.area;
			queue.push_back(nip); //�������
			color.at<int>(newx, newy) = 1; //�������
			//����alpha
			dands ds = vecds[nip.area];
			double alpha = Sigmoid((double)nip.dis / (double)stripwidth, delta(ds.delta), sigma(ds.sigma));
			alpha = adjustA(alpha);//����alpha
			bordermask.at<float>(nip.p.x, nip.p.y) = (float)alpha;
		}
	}
}

void BorderMatting::display(const Mat& oriImg, const Mat& borderMask)
{
	/*��mask���ִ���ǰ��*/
	vector<Mat> ch_img(3);
	vector<Mat> ch_bg(3);
	//����ǰ����ͨ��
	Mat img;
	oriImg.convertTo(img, CV_32FC3, 1.0 / 255.0);
	cv::split(img, ch_img);
	//���뱳����ͨ��
	Mat bg = Mat(img.size(), CV_32FC3, Scalar(1.0, 1.0, 1.0));
	cv::split(bg, ch_bg);
	//mask���ִ���
	ch_img[0] = ch_img[0].mul(borderMask) + ch_bg[0].mul(1.0 - borderMask);
	ch_img[1] = ch_img[1].mul(borderMask) + ch_bg[1].mul(1.0 - borderMask);
	ch_img[2] = ch_img[2].mul(borderMask) + ch_bg[2].mul(1.0 - borderMask);
	//�ϲ���ͨ��
	Mat res;
	cv::merge(ch_img, res);
	//��ʾ���
	Mat tem, tem2;
	resize(borderMask, tem2, Size(0, 0), 4, 4);
	resize(res, tem, Size(0, 0), 4, 4);
	imshow("result", tem);
	imshow("img", res);
	imshow("mask", tem2);
}

/*border matting*/
void BorderMatting::borderMatting(const Mat& oriImg, const Mat& mask, Mat& borderMask)
{
	/*��ʼ�����ֲ���*/
	init(oriImg);
	
	/*mask�������*/
	Mat edge = mask & 1;
	edge.convertTo(edge, CV_8UC1, 255);
	BorderDetection(edge,edge);
	
	/*����������*/
	ParameterizationContour(edge);
	
	/*��������*/
	Mat tmask;
	mask.convertTo(tmask,CV_8UC1);
	StripInit(tmask);

	/*������С������̬�滮��ÿ�������delta��sigma*/
	EnergyMinimization(oriImg, mask);

	/*�������� ���� alpha mask*/
	CalculateMask(borderMask, mask);	//����ÿ�����ص��alpha
	GaussianBlur(borderMask, borderMask, Size(7, 7), 9);	//��alpha mask������΢��˹ģ��

	/*��ʾborder matting���*/
	display(oriImg,borderMask);	
}


