#include "tpConnectedComponents.h"
#include <cmath>
#include <algorithm>
#include <tuple>
#include <vector>
#include <map>
#include <iostream>
#include <vector>
#include <stack>
using namespace cv;
using namespace std;


/**
    Performs a labeling of image connected component with 4 connectivity
    with a depth-first exploration.
    Any non zero pixel of the image is considered as present.
*/
void parcourCC(cv::Mat im, cv::Mat& res, int y, int x,int label){
    
    res.at<int>(y, x) = label;
    int directions[4][2] = {{-1, 0}, {1, 0}, {0, -1}, {0, 1}};

    for(auto dir: directions){
        int col = x + dir[0];
        int row = y + dir[1];
        if(row >= 0 && col >= 0 && row < im.rows && col < im.cols 
            && im.at<int>(row,col) != 0 
            && res.at<int>(row,col) == 0
           ){
                parcourCC(im,res,row,col,label);
        }
    }
}
cv::Mat ccLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int label = 1;
    for (int i = 0; i < image.rows; i++) {
        for (int j = 0; j < image.cols; j++) {
            if(image.at<int>(i,j)!=0 && res.at<int>(i,j)==0){
                parcourCC(image,res,i,j,label);
                label++;
            }
        }
    }
    /********************************************
                END OF YOUR CODE
    *********************************************/
    return res;
}

/**
    Deletes the connected components (4 connectivity) containg less than size pixels.
*/
cv::Mat ccAreaFilter(cv::Mat image, int size)
{
    Mat res = Mat::zeros(image.rows, image.cols, image.type());
    assert(size>0);
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int rows = image.rows;
    int cols = image.cols;

    vector<vector<bool>>visite(rows, vector<bool>(cols,false));
    vector<Point2i> neighbours = {{-1,0}, {0,-1}, {0,1}, {1,0}};
    stack<Point2i>pile;
    
    for(int y=0;y<rows;y++){
        for(int x=0;x<cols;x++){
     		if(!visite[y][x] && image.at<uchar>(y,x) != 0){
                vector<Point2i>component;
                pile.push(Point2i(x,y));
                component.push_back(Point2i(x,y));
                while(!pile.empty()){
                    Point2i r = pile.top();
                    pile.pop();

                    if(!visite[r.y][r.x]) continue;
                    visite[r.y][r.x] = true;
                    res.at<int>(r.y,r.x) = 1;
                    component.push_back(r);
        
                    for(Point2i &plus : neighbours){
                        Point2i v = r + plus;
                            if(v.x>=0 && v.y>=0 && v.x<cols && v.y<rows && !visite[v.y][v.x]){
                                pile.push(v);
                        }
                    }
                }
                int sizeCompo = component.size();
                if(sizeCompo >= size){
                    for(Point2i comp : component){
                        res.at<uchar>(comp.y,comp.x) = 255;
                    }
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
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/

bool contains(const vector<int>& a, int value) {
    return find(a.begin(), a.end(), value) != a.end();
}

vector<int> fusion(const vector<int>& a, const vector<int>& b) {
    vector<int> result = a;
    result.insert(result.end(), b.begin(), b.end());
    sort(result.begin(), result.end());
    result.erase(unique(result.begin(), result.end()), result.end());
    return result;
}

bool firstPass(Mat image, Mat res, Point2i p, int nb, vector<vector<int>> &equivalence) {
    if (image.at<int>(p.x, p.y) == 0) return false;
    if (p.x == 0 && p.y == 0) {
        res.at<int>(p.x, p.y) = nb;
        return true;
    }
    if (p.x == 0) {
        int enHaut = res.at<int>(p.x, p.y - 1);
        if (enHaut == 0 ) {
            res.at<int>(p.x, p.y) = nb;
            return true; 
        }
        else {
            res.at<int>(p.x, p.y) =  enHaut;
            return false;
        }

    }
    if (p.y == 0) {
        int gauche = res.at<int>(p.x - 1, p.y);
        if (gauche == 0) {
            res.at<int>(p.x, p.y) = nb;
            return true;
        } 
        else {
            res.at<int>(p.x, p.y) = gauche;
            return false;
        }
    }

    
    int enHaut = res.at<int>(p.x, p.y - 1);
    int gauche = res.at<int>(p.x - 1, p.y);

    if (enHaut == 0 && gauche == 0) {
         res.at<int>(p.x, p.y) = nb;
         return true;
    }
   
    if (enHaut == 0 && gauche != 0) {
         res.at<int>(p.x, p.y) = gauche;
         return false;
    }

    if (enHaut != 0 && gauche == 0) {
         res.at<int>(p.x, p.y) = enHaut;
         return false;
    }

    if (enHaut == gauche) {
        res.at<int>(p.x, p.y) = enHaut;
        return false;
    }

    int plusPetit = min(enHaut, gauche);
    int plusGros = max(enHaut, gauche);
    res.at<int>(p.x, p.y) = plusPetit;

    vector<int> jointure = {};
    for (int k = 0 ; k < (int) equivalence.size(); k++ ) {
        
         if (contains(equivalence[k], plusPetit) &&  contains(equivalence[k], plusGros)) {
            return false;
         }

        if (contains(equivalence[k], plusPetit) && ! contains(equivalence[k], plusGros)) {
            equivalence[k].push_back(plusGros);
            jointure.push_back(k);
        }
        if (contains(equivalence[k], plusGros) && ! contains(equivalence[k], plusPetit)) {            equivalence[k].push_back(plusPetit);
            jointure.push_back(k);
        }

    }

    if (jointure.size() > 1) { 
        equivalence[jointure[0]] = fusion(equivalence[jointure[0]], equivalence[jointure[1]]);
        equivalence.erase(equivalence.begin() + jointure[1]); 
    }
    if (jointure.size() == 0) {
        vector<int> zz;
        zz.push_back(plusPetit);
        zz.push_back(plusGros);
        equivalence.push_back(zz);
    }

    return false;
}

int minElement(vector<int> &e) {
    if (e.size() == 0) return 0;
    int res = e[0];
    for (int element: e) {
        if (element < res)
            res = element;
    }
    return res;
}

void secondPass(Mat& res, Point2i p, const vector<vector<int>>& equivalence) {
    if (res.at<int>(p.x, p.y) == 0) return;
    for (int k = 0; k < (int)equivalence.size(); k++) {
        if (contains(equivalence[k], res.at<int>(p.x, p.y))) {
            res.at<int>(p.x, p.y) = k + 1;
            return;
        }
    }
}

/**
    Performs a labeling of image connected component with 4 connectivity using a
    2 pass algorithm.
    Any non zero pixel of the image is considered as present.
*/
cv::Mat ccTwoPassLabel(cv::Mat image)
{
    Mat res = Mat::zeros(image.rows, image.cols, CV_32SC1); // 32 int image
    /********************************************
                YOUR CODE HERE
    *********************************************/
    int nb = 1;
    vector<vector<int>> equivalence = {};

    for(int y = 0; y < image.rows; y++) {     
        for (int x = 0 ; x < image.cols; x++) {
            if (firstPass(image, res, Point2i(y,x), nb, equivalence) ) {
                nb++;
            }
        }
    }

    for(int u = 1; u < nb; u++) {
        bool ajout = true;
        for (vector<int> equiv: equivalence) {
            if (contains(equiv, u)) { ajout = false; }
        }
        if (ajout) { equivalence.push_back({u}); }
    }
    
    for (int i = 0 ; i < (int)equivalence.size(); i++) {
        sort(equivalence[i].begin(), equivalence[i].end());
    }

    sort(equivalence.begin(), equivalence.end(),
        [](const std::vector<int>& a, const std::vector<int>& b) {
     return a[0] < b[0];
    });
    

    for (int i = 0 ; i < image.rows; i++) {
        for(int j = 0; j < image.cols; j++) {
            secondPass(res, Point2i(i,j), equivalence);   
        }
    }
    return res;
}

// Fait par CHEN Michel
