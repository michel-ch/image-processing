#include "tpGeometry.h"
#include <cmath>
#include <algorithm>
#include <tuple>
using namespace cv;
using namespace std;

/**
    Transpose the input image,
    ie. performs a planar symmetry according to the
    first diagonal (upper left to lower right corner).
*/
Mat transpose(Mat image)
{
    Mat res = Mat::zeros(image.cols,image.rows,CV_32FC1);
    /********************************************
                YOUR CODE HERE
    hint: consider a non square image
    *********************************************/
    int rows = image.rows;
    int cols = image.cols;
    
    
    for(int i=0;i<rows;i++){
    	for(int j=0;j<cols;j++){
    	    res.at<float>(j,i) = image.at<float>(i,j);
	}
    }
    
    

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Compute the value of a nearest neighbour interpolation
    in image Mat at position (x,y)
*/
float interpolate_nearest(Mat image, float y, float x)
{
    float v=0;
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = image.rows;
    int cols = image.cols;

    int close_x = round(x);
    int close_y = round(y);
    
    close_x = std::max(0,std::min(close_x,cols - 1));
    close_y = std::max(0,std::min(close_y, rows - 1));

    v = image.at<float>(close_y,close_x);

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return v;

}


/**
    Compute the value of a bilinear interpolation in image Mat at position (x,y)
*/
float interpolate_bilinear(Mat image, float y, float x)
{
    float v=0;
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = image.rows;
    int cols = image.cols;

    int cox = floor(x);
    int coy = floor(y);
    cox = std::max(0,std::min(cox, cols - 2));
    coy = std::max(0,std::min(coy, rows - 2));

    int voisin_x = cox + 1;
    int voisin_y = coy + 1;
    float xz = x - cox;
    float yz = y - coy;
    float tl = image.at<float>(coy,cox);
    float tr = image.at<float>(coy,voisin_x);
    float bl = image.at<float>(voisin_y,cox);
    float br = image.at<float>(voisin_y,voisin_x);
    
    float i1 = tl + (tr - tl) * xz;
    float i2 = bl + (br - bl) * xz;
    v = i1 + (i2 - i1) * yz;

    /********************************************
                END OF YOUR CODE
    *********************************************/
    return v;
}
/**
    Multiply the image resolution by a given factor using the given interpolation method.
    If the input size is (h,w) the output size shall be ((h-1)*factor, (w-1)*factor)
*/
Mat expand(Mat image, int factor, float(* interpolationFunction)(cv::Mat image, float y, float x))
{
    assert(factor>0);
    Mat res = Mat::zeros((image.rows-1)*factor,(image.cols-1)*factor,CV_32FC1);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = image.rows;
    int cols = image.cols;
    int rows2 = (rows - 1) * factor;
    int cols2 = (cols - 1) * factor;
    for (int i = 0; i < rows2; i++) {
        for (int j = 0; j < cols2; j++) {
            float base_y = (float)i / factor;
            float base_x = (float)j / factor;
            float interpolatedValue = interpolationFunction(image, base_y, base_x);
            res.at<float>(i, j) = interpolatedValue;
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Performs a rotation of the input image with the given angle (clockwise) and the given interpolation method.
    The center of rotation is the center of the image.

    Ouput size depends of the input image size and the rotation angle.

    Output pixels that map outside the input image are set to 0.
*/
Mat rotate(Mat image, float angle, float(* interpolationFunction)(cv::Mat image, float y, float x))
{
    /********************************************
                YOUR CODE HERE
    hint: to determine the size of the output, take
    the bounding box of the rotated corners of the 
    input image.
    *********************************************/
    int lignes = image.rows;
    int colonnes = image.cols;

    float angleRad = angle * M_PI / 180.0;

    float centreX = colonnes / 2.0;
    float centreY = lignes / 2.0;

    vector<Point2f> coins(4);
    coins[0] = Point2f(0, 0);              // Haut-gauche
    coins[1] = Point2f(colonnes - 1, 0);   // Haut-droit
    coins[2] = Point2f(colonnes - 1, lignes - 1); // Bas-droit
    coins[3] = Point2f(0, lignes - 1);     // Bas-gauche

    vector<Point2f> coinsTournes(4);
    for (int i = 0; i < 4; i++) {
        float x = coins[i].x - centreX;
        float y = coins[i].y - centreY;
        float x_tourne = x * cos(angleRad) - y * sin(angleRad);
        float y_tourne = x * sin(angleRad) + y * cos(angleRad);

        coinsTournes[i] = Point2f(x_tourne + centreX, y_tourne + centreY);
    }

    float min_x = min({coinsTournes[0].x, coinsTournes[1].x, coinsTournes[2].x, coinsTournes[3].x});
    float max_x = max({coinsTournes[0].x, coinsTournes[1].x, coinsTournes[2].x, coinsTournes[3].x});
    float min_y = min({coinsTournes[0].y, coinsTournes[1].y, coinsTournes[2].y, coinsTournes[3].y});
    float max_y = max({coinsTournes[0].y, coinsTournes[1].y, coinsTournes[2].y, coinsTournes[3].y});
    int nouvelles_colonnes = round(max_x - min_x);
    int nouvelles_lignes = round(max_y - min_y);

    Mat res = Mat::zeros(nouvelles_lignes, nouvelles_colonnes, CV_32FC1);

    float nouveauCentreX = nouvelles_colonnes / 2.0;
    float nouveauCentreY = nouvelles_lignes / 2.0;
    for (int i = 0; i < nouvelles_lignes; i++) {
        for (int j = 0; j < nouvelles_colonnes; j++) {
            float x = j - nouveauCentreX;
            float y = i - nouveauCentreY;
            float x_orig = x * cos(-angleRad) - y * sin(-angleRad);
            float y_orig = x * sin(-angleRad) + y * cos(-angleRad);

            x_orig += centreX;
            y_orig += centreY;

            if (x_orig >= 0 && x_orig < colonnes && y_orig >= 0 && y_orig < lignes) {
                float valeurInterpolee = interpolationFunction(image, y_orig, x_orig);
                res.at<float>(i, j) = valeurInterpolee;
            } else {
                res.at<float>(i, j) = 0.0;
            }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;

}
