#include <iostream>
#include "imgprocess.h"



int main() {

    Mat srcImg = imread("/Users/duanqifeng/CLionProjects/CarID/川O71775.jpg");
    Mat idImg = findIdImg(srcImg);
    vector<Mat> chars= getSingleChar(idImg);

    return 0;
}
