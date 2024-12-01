#include "tpHistogram.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <list>
using namespace cv;
using namespace std;

/**
    Inverse a grayscale image with float values.
    for all pixel p: res(p) = 1.0 - image(p)
*/
Mat inverse(Mat image)
{
    // clone original image
    Mat res = image.clone();

      /********************************************
                YOUR CODE HERE
    *********************************************/
    res = 1 - res;
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Thresholds a grayscale image with float values.
    for all pixel p: res(p) =
        | 0 if image(p) <= lowT
        | image(p) if lowT < image(p) <= hightT
        | 1 otherwise
*/
Mat threshold(Mat image, float lowT, float highT)
{
    Mat res = image.clone();
    assert(lowT <= highT);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for(int y = 0; y < res.rows; y++){
	for(int x = 0; x < res.cols; x++){
		float v = res.at<float>(y,x);
		if(v <= lowT){
			res.at<float>(y,x) = 0;
		}
		else if (lowT < v && v <= highT){
			v = v;
		}
		else{
			res.at<float>(y,x) = 1;
		}
	}
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Quantize the input float image in [0,1] in numberOfLevels different gray levels.
    
    eg. for numberOfLevels = 3 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/3
        | 1/2 if 1/3 <= image(p) < 2/3
        | 1 otherwise

        for numberOfLevels = 4 the result should be for all pixel p: res(p) =
        | 0 if image(p) < 1/4
        | 1/3 if 1/4 <= image(p) < 2/4
        | 2/3 if 2/4 <= image(p) < 3/4
        | 1 otherwise

       and so on for other values of numberOfLevels.

*/
Mat quantize(Mat image, int numberOfLevels)
{
    Mat res = image.clone();
    assert(numberOfLevels>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for(int y = 0; y < res.rows; y++){
        for(int x = 0; x < res.cols; x++){
                float v = res.at<float>(y,x);
		for( int level = 1; level<numberOfLevels; level++){
			if(v < level/numberOfLevels && level==1){
				res.at<float>(y,x) = 0;
				break;
			}
			else if(level-1/numberOfLevels <= v && v < level/numberOfLevels ){
				res.at<float>(y,x) = (level-1)/(numberOfLevels-1);
				break;
			}
			else{
				res.at<float>(y,x) = 1;
				break;
			}
		}
    	}
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Normalize a grayscale image with float values
    Target range is [minValue, maxValue].
*/
Mat normalize(Mat image, float minValue, float maxValue)
{
    Mat res = image.clone();
    assert(minValue <= maxValue);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    float fmin = image.at<float>(0,0);
    float fmax = image.at<float>(0,0);
    for(int y = 0; y < res.rows; y++){
        for(int x = 0; x < res.cols; x++){
		float v = res.at<float>(y,x);
		if( v < fmin ){
			fmin = v;
		}
		if( v > fmax ){
			fmax = v;
		}
	
	}
   }
    float ecart = maxValue - minValue;
    float ecartVieux = fmax - fmin;

    for(int y = 0; y < res.rows; y++){
        for(int x = 0; x < res.cols; x++){
                float v = res.at<float>(y,x);
		res.at<float>(y,x) = (v - fmin)*(ecart/ecartVieux)+minValue;
	}
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}



/**
    Equalize image histogram with unsigned char values ([0;255])

    Warning: this time, image values are unsigned chars but calculation will be done in float or double format.
    The final result must be rounded toward the nearest integer 
*/
Mat equalize(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = res.rows;
    int cols = res.cols;
    int pixels = rows*cols;
    int nbColor = 256;

    std::vector<int>nbpixels(nbColor,0);
    std::vector<float>histoCumule(nbColor,0.0f);
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
		int i = res.at<uchar>(y,x);
		nbpixels[i]++;
	}
    }
    
    histoCumule[0] = (float)nbpixels[0]/pixels;
    for(int i=1;i<nbColor;i++){
	histoCumule[i] = histoCumule[i-1] + (float)nbpixels[i]/ pixels;
    }

    float min = 0.0f;
    float max = 255.0f;

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
		int i = image.at<uchar>(y,x);
		float newPixel = (max - min)  * histoCumule[i] + min;
		res.at<uchar>(y,x) = (uchar)std::round(newPixel); 

	}
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;

}

/**
    Compute a binarization of the input float image using an automatic Otsu threshold.
    Input image is of type unsigned char ([0;255])
*/
Mat thresholdOtsu(Mat image)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = res.rows;
    int cols = res.cols;
    int pixels = rows*cols;
    int nbColor = 256;

    std::vector<int>histogram(nbColor,0);
    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
                int i = res.at<uchar>(y,x);
                histogram[i]++;
        }
    }

    float total = 0;
    for(int i=0;i<nbColor;i++){
	total += i * histogram[i];
    }

    float sumB = 0;
    int wB = 0;
    int wF = 0;
    float max = 0.0f;
    int threshold = 0;
    for(int i=1; i<nbColor; i++){
	wB += histogram[i];
	if(wB == 0) continue;
	wF = pixels - wB;
	if(wF == 0) break;

	sumB += i * histogram[i];
	float mB = sumB / wB;
	float mF = (total - sumB) / wF;
	float difference = wB * wF * (mB - mF) * (mB - mF);
	
	if(difference>max){
		max =  difference;
		threshold = i;
	}
    }

    for(int y = 0; y < rows; y++){
        for(int x = 0; x < cols; x++){
		if(res.at<uchar>(y,x) <= threshold){
			res.at<uchar>(y,x) = 0;
		}
		else{
			res.at<uchar>(y,x) = 255;
		}
	}
    }
	
    /*******************************************
                END OF YOUR CODE
    *********************************************/
    return res;

}

// Fait par CHEN Michel
