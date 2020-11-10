#include "../include/imgprocess.h"
#include "../include/svm.h"



int main() {

    Mat srcImg = imread("/Users/duanqifeng/CLionProjects/CarID/川A99999.jpg");

    Mat idImg = findIdImg(srcImg);
    vector<Mat> sigleIdimgs = getSigleImg(idImg);


    vector<Mat> ch,al,five;
    for (int i = 0; i < sigleIdimgs.size(); ++i) {
        if (i == 0){
            ch.push_back(sigleIdimgs[i]);
        } else if (i == 1){
            al.push_back(sigleIdimgs[i]);
        } else{
            five.push_back(sigleIdimgs[i]);
        }
    }

    cout<<"总数"<<sigleIdimgs.size()<<endl;
    cout<<ch.size()<<endl;
    cout<<al.size()<<endl;
    cout<<five.size()<<endl;

//    Ptr<SVM> model = trainSVM();//已经训练好了直接加载


    Ptr<SVM> model_ch = SVM::load("model_ch.xml");
    Ptr<SVM> model_al = SVM::load("model_al.xml");
    Ptr<SVM> model = SVM::load("model.xml");
    predict(ch,model_ch);
    predict(al,model_al);
    predict(five,model);

    return 0;
}




