#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/ml/ml.hpp>

namespace alfarobi_v8{

class AdaBoostWrapper{
public:
    AdaBoostWrapper(std::string&& _weight_path){
        boost_ = cv::ml::Boost::create();
        boost_ = cv::ml::Boost::load(_weight_path);
    }
    auto predict(const cv::Mat& _desc){
        return boost_->predict(_desc);
    }

private:
    cv::Ptr<cv::ml::Boost> boost_;
};

}