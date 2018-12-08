#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

int main() {
    Mat srcImg = imread("/Users/duanqifeng/CLionProjects/CarID/川A29922.jpg"),tgImg;
    resize(srcImg,srcImg,Size(640,480));
    Mat grayImg,blurImg,cannyImg,binimg;

    namedWindow("srcImg");
    namedWindow("tgImg");
    moveWindow("srcImg",0,0);
    moveWindow("tgImg",500,0);

    Mat element = getStructuringElement(MORPH_RECT,Size(3,3));

    cvtColor(srcImg,grayImg,CV_BGR2GRAY);
    blur(grayImg, blurImg, Size(3, 3));
    threshold(blurImg,blurImg,90,255,THRESH_BINARY_INV);
    imshow("bin",blurImg);
    Canny(blurImg,cannyImg,3,9,3);

    dilate(cannyImg,cannyImg,element);
    dilate(cannyImg,cannyImg,element);
//    morphologyEx(cannyImg,cannyImg,MORPH_GRADIENT,element);

    vector<vector<Point2i>> contours;
    findContours(cannyImg,contours,CV_RETR_LIST,CHAIN_APPROX_SIMPLE);
    for (int i = 0; i < contours.size(); ++i) {
        RotatedRect rect = minAreaRect(contours[i]);

        if ( rect.size.width > 60 && rect.size.height>10){
            cout<<rect.size<<endl;
            drawContours(srcImg,contours,i,Scalar(0,0,255));
            imshow("srcImg",srcImg);
            waitKey(0);
        }
    }


    imshow("srcImg",srcImg);
    imshow("tgImg",cannyImg);


    waitKey(0);
    return 0;
}