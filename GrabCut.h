#pragma once
#include <opencv2/opencv.hpp>
#include <iostream>
#include "BorderMatting.h"
#include "GMM.h"
#include "Bgraph.h"
using namespace std;
using namespace cv;

enum
{
	GC_WITH_RECT  = 0, 
	GC_WITH_MASK  = 1, 
	GC_CUT        = 2  
};
class GrabCut2D
{
public:
	void GrabCut( cv::InputArray _img, cv::InputOutputArray _mask, cv::Rect rect,
		cv::InputOutputArray _bgdModel,cv::InputOutputArray _fgdModel,
		int iterCount, int mode );  

	~GrabCut2D(void);
private:
	GMM bgdGMM,fgdGMM;
};

