#include "tpMorphology.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <limits>
#include "common.h"
using namespace cv;
using namespace std;


/**
    Compute a median filter of the input float image.
    The filter window is a square of (2*size+1)*(2*size+1) pixels.

    Values outside image domain are ignored.

    The median of a list l of n>2 elements is defined as:
     - l[n/2] if n is odd 
     - (l[n/2-1]+l[n/2])/2 is n is even 
*/
Mat median(Mat image, int size)
{
    Mat res = image.clone();
    assert(size>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    for (int i = 0; i < image.rows; ++i) {
        for (int j = 0; j < image.cols; ++j) {
            vector<float> list;
            for (int m = -size; m <= size; ++m) {
                for (int n = -size; n <= size; ++n) {
                    int x = i + m;
                    int y = j + n;
                    if (x >= 0 && x < image.rows && y >= 0 && y < image.cols) {
                        list.push_back(image.at<float>(x, y));
                    }
                }
            }
            sort(list.begin(), list.end());
            int mid = list.size() / 2;
            float median;
            if (list.size() % 2 == 0) {
                median = (list[mid - 1] + list[mid]) / 2.0f;
            } else {
                median = list[mid];
            }
            res.at<float>(i, j) = median;
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the dilation of the input float image by the given structuring element.
     Pixel outside the image are supposed to have value 0
*/
Mat dilate(Mat image, Mat structuringElement)
{
    /********************************************
                YOUR CODE HERE
    *********************************************/
    Mat res = Mat::zeros(image.size(), CV_32FC1);
    int rows = image.rows;
    int cols = image.cols;
    int sRows = structuringElement.rows;
    int sCols = structuringElement.cols;
    int CenterX = sRows / 2;
    int CenterY = sCols / 2;

    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            float maxValeur = 0.0f;
            for (int m = 0; m < sRows; ++m) {
                for (int n = 0; n < sCols; ++n) {
                    int x = i + m - CenterX;
                    int y = j + n - CenterY;
                    if (structuringElement.at<uchar>(m, n) != 0) {
                        if (x >= 0 && x < rows && y >= 0 && y < cols) {
                            maxValeur = max(maxValeur, image.at<float>(x, y));
                        }
                    }
                }
            }
            res.at<float>(i, j) = maxValeur;
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the erosion of the input float image by the given structuring element.
    Pixel outside the image are supposed to have value 1.
*/
Mat erode(Mat image, Mat structuringElement)
{
    Mat res = image.clone();
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
    if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U, 255);
    }
    if (structuringElement.type() != CV_8U) {
        structuringElement.convertTo(structuringElement, CV_8U);
    }
    erode(image, res, structuringElement, Point(-1, -1), 1, BORDER_CONSTANT, Scalar(1));
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the opening of the input float image by the given structuring element.
*/
Mat open(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
   if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U, 255);
    }
    if (structuringElement.type() != CV_8U) {
        structuringElement.convertTo(structuringElement, CV_8U);
    }
    morphologyEx(image, res, MORPH_OPEN, structuringElement);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the closing of the input float image by the given structuring element.
*/
Mat close(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
   if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U, 255);
    }
    if (structuringElement.type() != CV_8U) {
        structuringElement.convertTo(structuringElement, CV_8U);
    }
    morphologyEx(image, res, MORPH_CLOSE, structuringElement);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}


/**
    Compute the morphological gradient of the input float image by the given structuring element.
*/
Mat morphologicalGradient(Mat image, Mat structuringElement)
{

    Mat res = Mat::zeros(1,1,CV_32FC1);
    /********************************************
                YOUR CODE HERE
        hint : 1 line of code is enough
    *********************************************/
   if (image.type() != CV_8U) {
        image.convertTo(image, CV_8U, 255);
    }
    if (structuringElement.type() != CV_8U) {
        structuringElement.convertTo(structuringElement, CV_8U);
    }
    morphologyEx(image, res, MORPH_GRADIENT, structuringElement);
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

