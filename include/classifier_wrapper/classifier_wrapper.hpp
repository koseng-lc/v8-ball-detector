#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

namespace alfarobi_v8{

class ClassifierWrapper{
public:
    ClassifierWrapper(std::string&& _weight_path){
        svm_ = cv::ml::SVM::create();
        svm_ = cv::ml::SVM::load(_weight_path);
    }
    auto predict(const cv::Mat& _desc){
        return svm_->predict(_desc);
    }

private:
    cv::Ptr<cv::ml::SVM> svm_;
};

}