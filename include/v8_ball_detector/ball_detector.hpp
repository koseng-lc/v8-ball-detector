#pragma once

#include <iostream>

#include <boost/chrono.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace alfarobi_v8{

using default_scalar_t = double;
constexpr std::size_t default_width{640};
constexpr std::size_t default_height{480};

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width = default_width, std::size_t height = default_height>
class BallDetector{
public:
    BallDetector(Search* _search, Descriptor* _desc, Classifier* _classifier)
        : search_(_search)
        , desc_(_desc)
        , classifier_(_classifier){

    }

    auto execute(const cv::Mat&  _in);

private:
    auto loadLUT(std::string&& _file_path);
    auto segmentField(const cv::Mat& _in);

private:
    enum class ColorFlag{
        Field=1,
        Ball=2
    };

private:
    Search* search_;
    Descriptor* desc_;
    Classifier* classifier_;    

    static constexpr auto hue_range_{180};
    static constexpr auto sat_range_{255};
    std::array<default_scalar_t, hue_range_*sat_range_> field_lut_;
};

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::loadLUT(std::string&& _file_path){
    
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::segmentField(const cv::Mat& _in_hsv){
    cv::Mat field(cv::Mat::zeros(height, width, CV_8UC1));
    
    for(auto i(0); i < _in_hsv.total(); i++){
        int hue = _in_hsv.at<cv::Vec3b>(i)[0];
        int sat = _in_hsv.at<cv::Vec3b>(i)[1];                
        int val = _in_hsv.at<cv::Vec3b>(i)[2];

        if(field_lut_[hue*sat_range_ + sat] == ColorFlag::Field)
            field.at<uchar>(i) = 255;
    }

    return field;
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::execute(const cv::Mat&  _in){
    auto t1 = boost::chrono::high_resolution_clock::now();

    cv::Mat input_hsv;
    cv::cvtColor(_in, input_hsv, cv::COLOR_BGR2HSV);

    cv::imshow("[alfarobi_v8][BallDetector] Input as HSV", input_hsv);

    auto t2 = boost::chrono::high_resolution_clock::now();

    auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();        
    std::cout << "[alfarobi_v8][BallDetector] Elapsed time: "<< (default_scalar_t)elapsed_time << " ms" << std::endl;
}

}