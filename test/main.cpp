#include "imgProcess.h"

using namespace std;
using namespace cv;

int main() {

    Mat src = imread("/Users/duanqifeng/CLionProjects/duanxiaoer/sample/P90101-143404.jpg");
    resize(src,src,Size(480,640));
    Mat grayImg;
    cvtColor(src,grayImg,COLOR_BGR2GRAY);
    Mat binaryImg;
    threshold(grayImg,binaryImg,100,250,THRESH_BINARY);

    imshow("src",src);
    imshow("gray",grayImg);
    imshow("binary",binaryImg);
    waitKey(-1);
    return 0;
}