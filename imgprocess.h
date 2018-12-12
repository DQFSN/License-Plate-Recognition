//
// Created by 段其沣 on 2018/12/11.
//

#ifndef CARID_IMGPROCESS_H
#define CARID_IMGPROCESS_H

#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

#define IDIMGROWS 35
#define IDIMGCOLS 110

//#define DETAIL
#define PRINTDETAIL
#define IDIMGDETAIL

Mat findIdImg(Mat carImg);

vector<Mat> getSingleChar(const Mat &idImg);

string getID(vector<Mat>);

Mat findIdImg(Mat carImg){
    resize(carImg,carImg,Size(640,480));

    namedWindow("srcImg");
    namedWindow("cannyImg");
    namedWindow("hsv");
    moveWindow("srcImg",0,0);
    moveWindow("hsv",600,600);
    moveWindow("cannyImg",650,0);

    Mat idImg;//返回的车牌截图
    Mat hsvImg;
    cvtColor(carImg,hsvImg,CV_BGR2HSV);

    enum colorType {BLUE=0,YELLOW,WHITE};
    const Scalar blueLow = Scalar(90,80,80);
    const Scalar blueHi = Scalar(120,220,255);
    const Scalar yellowLo = Scalar(26,43,46);
    const Scalar yellowHi = Scalar(34,255,255);
    const Scalar whiteLo = Scalar(0,0,211);
    const Scalar whiteHi = Scalar(180,255,46);
    vector<Scalar> hsvLo = {blueLow,yellowLo,whiteLo};
    vector<Scalar> hsvHi = {blueHi,yellowHi,whiteHi};

    inRange(hsvImg,hsvLo[0],hsvHi[0],hsvImg);//找图片中复合颜色范围的区域，不符合该点像素值为0

#ifdef DETAIL
    imshow("hsv",hsvImg);
#endif

    Mat erode_dilate = hsvImg;//膨胀和腐蚀
    Mat element = getStructuringElement(MORPH_RECT,Size(3,3));
    dilate(erode_dilate,erode_dilate,element);
    dilate(erode_dilate,erode_dilate,element);
//    erode(erode_dilate,erode_dilate,element);
//    dilate(erode_dilate,erode_dilate,element);

#ifdef DETAIL
    imshow("erode/dilate",erode_dilate);
#endif

    Mat cannyImg;
    Canny(erode_dilate,cannyImg,3,9);

#ifdef DETAIL
    imshow("canny",cannyImg);
#endif

    vector<vector<Point2i>> contours;
    findContours(cannyImg,contours,RETR_EXTERNAL,CHAIN_APPROX_NONE);

    int templateH = 40;//templateW = 160   //此处推荐采用height
    double lowR = 0.5,highR = 2;//车牌高度和模板的比值
    double low_WH = 2,high_WH = 6;//车牌width和height的比值
//    for (int index=0;index<contours.size();++index){
    for (const auto &contour : contours) {
        RotatedRect rect = minAreaRect(contour);
        float rectAngel = rect.angle;
        int rectW,rectH;
        if (rectAngel > -10){
            rectW = (int)rect.size.width;
            rectH = (int)rect.size.height;
        } else if ( rectAngel < -80 ){
            rectAngel = 90+rectAngel;//为了之后修正图像时旋转角度正确
            rectW = (int)rect.size.height;
            rectH = (int)rect.size.width;
        } else{
            continue;
        }

        double ratio = rectW*1.0 / rectH;
        if (rectH > templateH * lowR && rectH < templateH * highR && ratio >low_WH && ratio < high_WH && rect.center.y>200){
#ifdef DETAIL
            cout<<rect.size<<", "<<rectW<<" X "<<rectH<<", angle:"<<rect.angle<<", center:"<<rect.center<<endl;
            Point2f points[4];
            rect.points(points);
            for (int i = 0; i < 4; ++i) {
                line(carImg,points[i],points[(i+1)%4],Scalar(0,255,0));
            }
#endif

            //修正图像并截取
            Point center = rect.center;
            Mat rot = getRotationMatrix2D(center,rectAngel,1);
            Mat src_copy_roi;
            warpAffine(carImg,src_copy_roi,rot,Size(carImg.cols,carImg.rows));

#ifdef DETAIL
            imshow("roi",src_copy_roi);//旋转后的图像
#endif
            Rect rect_roi(center.x - rectW/2,center.y-rectH/2,rectW,rectH);
            idImg = src_copy_roi(rect_roi);//从旋转后的图像上截取车牌

            return idImg;//返回车牌图片

#ifdef DETAIL
            imshow("cannyImg",idImg);
            waitKey(0);
#endif
        }
    }

    cerr<<"没找到车牌，请重试！"<<endl;
    exit(1);
}

vector<Mat> getSingleChar(const Mat &idImg){
    Mat src = idImg.clone();
    resize(src,src,Size(IDIMGCOLS,IDIMGROWS));
    Mat grayImg;
    cvtColor(src,grayImg,COLOR_BGR2GRAY);
    Mat cannyImg;
    Canny(grayImg,cannyImg,90,120);
    Mat binImg;
    threshold(cannyImg,binImg,130,255,THRESH_OTSU);

    int firstRow = 0,lastRow = 0;

    //水平扫描
    int leapRowDot[IDIMGROWS] = {0};//每行跳跃点个数
    int leapColDot[IDIMGCOLS] = {0};//每列跳跃点个数
    for (int rowIndex=0; rowIndex < binImg.rows-1; ++rowIndex) {
        for (int colIndex = 0; colIndex < binImg.cols-1; ++colIndex) {
            //行跳跃点
            Point2i point_l(colIndex,rowIndex);
            Point2i point_r(colIndex+1,rowIndex);
            bool value_l = binImg.at<uchar>(point_l);
            bool value_r = binImg.at<uchar>(point_r);
            if ( value_l != value_r)
                ++leapRowDot[rowIndex];

#ifdef PRINTDETAIL
            cout<<binImg.at<uchar>(point_l);
#endif

        }

        //寻找第一行
        if ( rowIndex > 2 && rowIndex < IDIMGROWS/2 && leapRowDot[rowIndex] < 30){
            firstRow = rowIndex;
        }
        //寻找最后一行
        if (rowIndex > IDIMGROWS/2 && leapRowDot[rowIndex] > 20 && leapRowDot[rowIndex-1] >20 && leapRowDot[rowIndex-2] > 20){
            lastRow = rowIndex;
        }


#ifdef PRINTDETAIL
        cout<<":"<<leapRowDot[rowIndex]<<endl;
#endif
    }

#ifdef PRINTDETAIL
    cout<<"\n\n\n"<<endl;
#endif


    //输出去掉多余行后的图，并统计列跳跃点
    for (int rowIndex=firstRow; rowIndex < lastRow+1; ++rowIndex) {
        for (int colIndex = 0; colIndex < binImg.cols-1; ++colIndex) {
            //列跳跃点
            Point2i point_u(colIndex,rowIndex);
            Point2i point_d(colIndex,rowIndex+1);
            bool value_u = binImg.at<uchar>(point_u);
            bool value_d = binImg.at<uchar>(point_d);
            if (value_u != value_d)
                ++leapColDot[colIndex];

#ifdef PRINTDETAIL
            cout<<binImg.at<uchar>(point_u);
#endif
        }
#ifdef PRINTDETAIL
        cout<<":"<<leapRowDot[rowIndex]<<endl;
#endif
    }

    //寻找分割列
    bool firstSplitCol = true;
    int colSplitCols[8] = {0};//分割列的索引
    int preColNum = 0;//目前是第几根分割列
    //输出列跳跃点
    for(int i = 0;i<IDIMGCOLS ;++i){
#ifdef PRINTDETAIL
        cout<<leapColDot[i]/10;
#endif

        int floor = 4;//判定为分割线的阈值
        //找到分割数字的列
        if (firstSplitCol){
            if (i>16 && leapColDot[i]<floor){
                colSplitCols[preColNum] = i;
                ++preColNum;
                firstSplitCol = false;
            }
        } else{
            int colStep = 12;//分割线彼此的距离
            if (preColNum == 2 ){
                colStep = 5;
            }
            if (i-colSplitCols[preColNum-1] > colStep && leapColDot[i]<floor){
                colSplitCols[preColNum] = i;
                ++preColNum;
            }
        }

    }

    if (preColNum < 8 && lastRow<=firstRow){
        cerr<<"列切割失败"<<endl;
        exit(-1);
    }

#ifdef PRINTDETAIL
    cout<<endl;

    for (auto f : leapColDot){
        cout<<f%10;
    }

    cout<<endl;
#endif




#ifdef IDIMGDETAIL

    //画出列跳跃点分布
    Mat plate_x = Mat(Size(110,35),CV_8UC3,Scalar(255,255,255));
    for (int i = 0; i < IDIMGCOLS; ++i) {
        line(plate_x,Point2i(i,110),Point2i(i,35-leapColDot[i]),Scalar(0,0,255));
    }
    //画出行跳跃点分布
    Mat plate_y = Mat(Size(110,35),CV_8UC3,Scalar(255,255,255));
    for (int i = 0; i < IDIMGROWS; ++i) {
        line(plate_y,Point2i(0,i),Point2i(leapRowDot[i],i),Scalar(0,0,255));
    }



    for (auto tmp : colSplitCols){
        line(plate_x,Point2i(tmp,35),Point2i(tmp,0),Scalar(0,0,255));

#ifdef PRINTDETAIL
        cout<<"colSplits: "<<tmp<<" ";
#endif
    }


#ifdef PRINTDETAIL
    cout<<endl;
#endif

    imshow("line_X",plate_x);
    imshow("line_y",plate_y);
    imshow("binIDImg",binImg);
    waitKey(0);

#endif

    Rect rect(0,firstRow,IDIMGCOLS,lastRow-firstRow+1);
    Mat new_binImg = binImg(rect);
    cout<<"new_binimg.size: "<<new_binImg.size<<endl;

    vector<Mat> singleCharImg = {new_binImg};
    int begin = 0;
    for (int colSplitCol : colSplitCols) {
        int width = colSplitCol - begin + 1;
        int height = lastRow-firstRow+1;
        Rect rect_single(begin+2,0,width-2,height);//+2,因为切片出来的效果感觉偏左，所以右移了一点
        singleCharImg.push_back(new_binImg(rect_single));
        begin = colSplitCol;
    }

#ifdef IDIMGDETAIL
    int NUM = 0;
    for (auto const &tmp : singleCharImg){
        string windowName = to_string(NUM) +"th";
        imshow(windowName,tmp);
        waitKey(0);
        ++NUM;
    }
#endif
    return singleCharImg;
}


#endif //CARID_IMGPROCESS_H
