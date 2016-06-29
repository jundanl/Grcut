#include <iostream>
#include "GCApplication.h"
static void help()
{
	std::cout << "\nThis program demonstrates GrabCut segmentation -- select an object in a region\n"
		"and then grabcut will attempt to segment it out.\n"
		"Call:\n"
		"./grabcut <image_name>\n"
		"\nSelect a rectangular area around the object you want to segment\n" <<
		"\nHot keys: \n"
		"\tESC - quit the program\n"
		"\tr - restore the original image\n"
		"\tn - next iteration\n"
		"\n"
		"\tleft mouse button - set rectangle\n"
		"\n"
		"\tCTRL+left mouse button - set GC_BGD pixels\n"
		"\tSHIFT+left mouse button - set CG_FGD pixels\n"
		"\n"
		"\tCTRL+right mouse button - set GC_PR_BGD pixels\n"
		"\tSHIFT+right mouse button - set CG_PR_FGD pixels\n" << endl;
}


GCApplication gcapp;

static void on_mouse( int event, int x, int y, int flags, void* param )
{
	gcapp.mouseClick( event, x, y, flags, param );
}


int main()
{
	cout << "Input image path please:" << endl;
	string filename;
	cin >> filename;
	Mat image = imread( filename, 1 );
	if( image.empty() )
	{
		cout << "\n , couldn't read image filename " << filename << endl;
		return 1;
	}

	help();

	const string winName = "image";
	cvNamedWindow( winName.c_str(), CV_WINDOW_AUTOSIZE );
	cvSetMouseCallback( winName.c_str(), on_mouse, 0 );

	gcapp.setImageAndWinName( image, winName );
	gcapp.showImage();

	for(;;)
	{
		int c = cvWaitKey(0);
		int iterCount;
		switch( (char) c )
		{
		case '\x1b':
			cout << "Exiting ..." << endl;
			goto exit_main;
		case 'r':
			cout << (char)c <<endl;
			gcapp.reset();
			gcapp.showImage();
			break;
		case 'b':
			cout << (char)c << endl;
			iterCount = gcapp.getIterCount();
			if (iterCount == 0)
				cout << "Please implement grabcut first." << endl;
			else
			{
				gcapp.borderMatting();
				gcapp.showImage();
			}
			break;
		case 'n':
			cout << (char)c << endl;
			iterCount = gcapp.getIterCount();
			cout << "<" << iterCount << "... ";
			int newIterCount = gcapp.nextIter();
			if (newIterCount > iterCount)
			{
				gcapp.showImage();
				cout << iterCount << ">" << endl;
			}
			else
				cout << "rect must be determined>" << endl;
			break;
		}
	}

	exit_main:
	cvDestroyWindow( winName.c_str() );
	return 0;

	return 0;
}