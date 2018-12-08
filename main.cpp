#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
    namedWindow("srcImg");
    namedWindow("tgImg");
    moveWindow("srcImg",0,0);
    moveWindow("tgImg",0,500);

    Mat srcImg = imread("/Users/duanqifeng/CLionProjects/CarID/鄂A59770.jpg"),tgImg;
    resize(srcImg,srcImg,Size(640,480));

    Mat hsvImg;
    cvtColor(srcImg,hsvImg,CV_BGR2HSV);

    enum colorType {BLUE=0,YELLOW,WHITE};
    const Scalar blueLow = Scalar(100,43,46);
    const Scalar blueHi = Scalar(124,255,255);
    const Scalar yellowLo = Scalar(26,43,46);
    const Scalar yellowHi = Scalar(34,255,255);
    const Scalar whiteLo = Scalar(0,0,211);
    const Scalar whiteHi = Scalar(180,255,46);

    vector<Scalar> hsvLo = {blueLow,yellowLo,whiteLo};
    vector<Scalar> hsvHi = {blueHi,yellowHi,whiteHi};

    inRange(hsvImg,hsvLo[0],hsvHi[0],hsvImg);
    Mat binImg;
    threshold(hsvImg,binImg,1,255,THRESH_BINARY);
    Mat cannyImg;
    Canny(binImg,cannyImg,3,9);

    vector<vector<Point2i>> contours;
    findContours(binImg,contours,RETR_LIST,CHAIN_APPROX_NONE);

    for (int index=0;index<contours.size();++index){
        drawContours(srcImg,contours,index,Scalar(0,0,255));
    }

    imshow("srcImg",srcImg);
    imshow("tgImg",binImg);

    waitKey(0);
    return 0;
}