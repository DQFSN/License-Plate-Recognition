//
// Created by 段其沣 on 2018/12/30.
//

#ifndef CARID_SVM_H
#define CARID_SVM_H

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <sys/types.h>
#include <fcntl.h>
#include <dirent.h>
#include <vector>
#include <sstream>
using namespace cv;
using namespace std;
using namespace ml;

vector<Mat> trainingImages; //用来存放训练图像信息的容器
vector<int> trainingLabels; //用来存放图像对应正负样本的值，正样本为1，负样本为0

int one_hot[65]{0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64};
int one_hot_al_num[34]{0,1,2,3,4,5,6,7,8,9,41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64};
int one_hot_al[24]{41,42,43,44,45,46,47,48,49,50,51,52,53,54,55,56,57,58,59,60,61,62,63,64};
int one_hot_ch[31]{10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31,32,33,34,35,36,37,38,39,40};

string labels[65] = {"0","1","2","3","4","5","6","7","8","9","藏","川","鄂","甘","赣","贵","桂","黑","沪","吉","冀","津","晋","京","辽","鲁", "蒙","闽","宁","青","琼","陕","苏","皖","湘","新","渝","豫","粤","云","浙","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};
string labels_al_num[34] = {"0","1","2","3","4","5","6","7","8","9","A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};
string labels_al[24] = {"A","B","C","D","E","F","G","H","J","K","L","M","N","P","Q","R","S","T","U","V","W","X","Y","Z"};
string labels_ch[31] = {"藏","川","鄂","甘","赣","贵","桂","黑","沪","吉","冀","津","晋","京","辽","鲁", "蒙","闽","宁","青","琼","陕","苏","皖","湘","新","渝","豫","粤","云","浙"};

int sampleNum[65] = {2574,1806,2932,2739,2183,3131,3068,2775,3093,3080,19,184,123,105,117,93,83,153,223,83,123,329,129,197,127,148,96,97,62,77,61,128,269,245,194,54,95,218,222,65,307,1466,1353,727,732,660,665,693,629,813,823,616,606,601,644,576,668,3600,506,580,586,526,656,794,559};
int sampleNum_al_num[34] = {2574,1806,2932,2739,2183,3131,3068,2775,3093,3080,1466,1353,727,732,660,665,693,629,813,823,616,606,601,644,576,668,3600,506,580,586,526,656,794,559};
int sampleNum_al[24] = {1466,1353,727,732,660,665,693,629,813,823,616,606,601,644,576,668,3600,506,580,586,526,656,794,559};
int sampleNum_ch[31] = {19,184,123,105,117,93,83,153,223,83,123,329,129,197,127,148,96,97,62,77,61,128,269,245,194,54,95,218,222,65,307};


void openfile(int flag, const string &dpath){//获取图像的路径

    //将图像对应的值存放到容器中
    Mat img = imread(dpath);
    resize(img,img,Size(18,27));
//    imshow("aa",img);
//    waitKey(0);
    Mat line_i = img.reshape(1, 1);

    trainingImages.push_back(line_i);
    trainingLabels.push_back(flag);
}

Ptr<SVM> trainSVM(){
    Mat classes;

    stringstream s;
    string numAl_basePath = "/Users/duanqifeng/Downloads/兰志城/车牌字符集/all/";
//    /Users/duanqifeng/Downloads/车牌字符集/all/0/0_0.jpg

    int index = 0;
    //加载训练图片
    for (const auto &l : labels_al) {
        for (int i=0;i<sampleNum_al[index];++i){
            s<<numAl_basePath<<l<<"/"<<l<<"_"<<i<<".jpg";
            string r;
            s>>r;
            s.clear();
            openfile(one_hot_al[index],r);
        }
        ++index;
    }

    //训练图片转换为训练数据
    Mat trainingData(static_cast<int>(trainingImages.size()), trainingImages[0].cols, CV_32FC1);
    for (int i = 0; i < trainingImages.size(); i++)
    {
        Mat temp(trainingImages[i]);
        temp.copyTo(trainingData.row(i));
    }
    trainingData.convertTo(trainingData, CV_32FC1);
    Mat(trainingLabels).copyTo(classes);
    classes.convertTo(classes, CV_32SC1);


    //以下是设置SVM训练模型的配置
    Ptr<SVM> model = SVM::create();
    model->setType(SVM::C_SVC);
    model->setKernel(SVM::LINEAR);
    model->setGamma(1);
    model->setC(1);
    model->setCoef0(0);
    model->setNu(0);
    model->setP(0);
    model->setTermCriteria(cvTermCriteria(CV_TERMCRIT_ITER, 20000, 0.0001));


    //开始训练模型
    Ptr<TrainData> tdata = TrainData::create(trainingData, ROW_SAMPLE, classes);
    model->train(tdata);
    model->save("model_al.xml");


    return model;
}

void predict(vector<Mat> imgs, const Ptr<SVM> &model){//type值含义： 0->汉字；1->字母；2->数字和字母

    vector<Mat> testImages; //用来存放测试图像信息的容器

    for (auto img : imgs) {
        resize(img,img,Size(18,27));
        Mat line_i = img.reshape(1, 1);
        testImages.push_back(line_i);
    }

    //测试图片转换为测试数据
    Mat testData(static_cast<int>(testImages.size()), testImages[0].cols, CV_32FC1);
    for (int i = 0; i < testImages.size(); i++)
    {
        Mat temp(testImages[i]);
        temp.copyTo(testData.row(i));
    }
    testData.convertTo(testData, CV_32FC1);
    //预测
    string result[10];
    for (int j = 0; j < testData.rows; ++j) {
        int index = static_cast<int>(model->predict(testData.row(j)));

        result[j] = labels[index];

        cout<<result[j];
    }
}




#endif //CARID_SVM_H
