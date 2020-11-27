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
    static constexpr auto hue_range_{180};
    static constexpr auto sat_range_{256};

    BallDetector(Search* _search, Descriptor* _desc, Classifier* _classifier)
        : search_(_search)
        , desc_(_desc)
        , classifier_(_classifier)
        , color_table_(hue_range_, sat_range_, CV_8UC1){

    }

    auto loadConfig(std::string&& _file_path);
    auto execute(const cv::Mat&  _in);

private:    
    auto segmentField(const cv::Mat& _in_hsv);
    auto segmentBall(const cv::Mat& _in_hsv);
    auto cropField(const cv::Mat& _field_color, cv::Mat* _field_contour);

private:
    struct BallParam{
        int min_val{0};
        int max_val{255};
        int min_sat{0};
        int max_sat{255};
    }ball_param_;

    enum class ColorFlag{
        None=0,
        Field=1,
        Ball=2
    };

private:
    Search* search_;
    Descriptor* desc_;
    Classifier* classifier_;    
    
    cv::Mat color_table_;

    static constexpr auto MIN_FIELD_PART_CONTOUR_AREA{800.};
    static constexpr auto MIN_MAYBE_BALL_CONTOUR_AREA{100.};
};

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::loadConfig(std::string&& _file_path){
    cv::FileStorage fs("../data/v8_ball_detector_config.yaml", cv::FileStorage::READ);
    fs["color_table"] >> color_table_;
    fs["min_val"] >> ball_param_.min_val;
    fs["max_val"] >> ball_param_.max_val;
    fs["min_sat"] >> ball_param_.min_sat;
    fs["max_sat"] >> ball_param_.max_sat;
    fs.release();
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::segmentField(const cv::Mat& _in_hsv){
    cv::Mat field(cv::Mat::zeros(height, width, CV_8UC1));
    
    for(auto i(0); i < _in_hsv.total(); i++){
        int hue = _in_hsv.at<cv::Vec3b>(i)[0];
        int sat = _in_hsv.at<cv::Vec3b>(i)[1];                
        int val = _in_hsv.at<cv::Vec3b>(i)[2];

        if(color_table_.at<uchar>(hue*sat_range_ + sat) == (uchar)ColorFlag::Field)
            field.at<uchar>(i) = 255;
    }

    return field;
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::segmentBall(const cv::Mat& _in_hsv){
    cv::Mat ball_color(cv::Mat::zeros(_in_hsv.size(), CV_8UC1));
    
    for(auto i(0); i < _in_hsv.total(); i++){
        int sat = _in_hsv.at<cv::Vec3b>(i)[1];                
        int val = _in_hsv.at<cv::Vec3b>(i)[2];

        if(val >= ball_param_.min_val & sat >= ball_param_.min_sat
            & val <= ball_param_.max_val & sat <= ball_param_.max_sat)
            ball_color.at<uchar>(i) = 255;
    }

    return ball_color;
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::cropField(const cv::Mat& _field_color, cv::Mat* _field_contour){
    *_field_contour = cv::Mat::zeros(_field_color.size(), CV_8UC1);
    std::vector<cv::Point> unified_contour;    
    std::vector<std::vector<cv::Point>> contours;
    std::vector<cv::Vec4i> hierarchy;

    cv::findContours(_field_color, contours, hierarchy, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

    for(std::size_t i(0); i < contours.size(); i++){
        if((cv::contourArea(contours[i]) > MIN_FIELD_PART_CONTOUR_AREA)){
        	unified_contour.insert(unified_contour.end(), contours[i].begin(), contours[i].end());    
        }
    }

    if(unified_contour.size()){        
        std::vector<std::vector<cv::Point>> temp(1);
        cv::convexHull(unified_contour, temp[0]);
        cv::drawContours(*_field_contour, temp, 0, cv::Scalar(255), cv::FILLED);
    }
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::execute(const cv::Mat&  _in){
    auto t1 = boost::chrono::high_resolution_clock::now();
    
    cv::Mat input_hsv;
    cv::Mat field_color;
    cv::Mat ball_color;
    cv::Mat field_contour;
    cv::Mat cropped_invert_field;
    cv::cvtColor(_in, input_hsv, cv::COLOR_BGR2HSV);
    
    field_color = segmentField(input_hsv);
    ball_color = segmentBall(input_hsv);
    cropField(field_color, &field_contour);

    cv::multiply(~field_color, field_contour, cropped_invert_field);
    std::vector<std::vector<cv::Point>> contours;
    cv::findContours(ball_color, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);
    cv::Mat temp_canvas(cv::Mat::zeros(height, width, CV_8UC1));
	for(std::size_t i(0); i < contours.size(); i++){
		if(cv::contourArea(contours[i]) > MIN_MAYBE_BALL_CONTOUR_AREA){
			std::vector<std::vector<cv::Point>> temp(1);
			cv::convexHull(contours[i], temp[0]);
			cv::drawContours(temp_canvas, temp, 0, cv::Scalar(255), cv::FILLED);
		}
	}	
    cv::Mat temp(cropped_invert_field.clone());
    // std::vector<std::vector<cv::Point>> contours;
    contours.clear();
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(cropped_invert_field, contours, hierarchy,
                        cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);    
    std::vector<double> contour_area;
    for(std::size_t i(0); i < contours.size(); i++){
    	contour_area.emplace_back(cv::contourArea(contours[i]));
        if(contour_area[i] > MIN_MAYBE_BALL_CONTOUR_AREA){       	
        	if(hierarchy[i][3] == -1){
        		cv::drawContours(temp, contours, i, cv::Scalar(255), cv::FILLED);
        	}
        }
    }  

    for(std::size_t i(0); i < contours.size(); i++){    	
        if(contour_area[i] > 500.){ //-- magic number   	
        	if(hierarchy[i][3] > -1){
        		cv::drawContours(temp, contours, i, cv::Scalar(0), cv::FILLED);
        	}
        }
    }
    cv::Mat ball_candidate_contours;
    cv::bitwise_and(temp, temp_canvas, ball_candidate_contours);

    

    cv::imshow("[alfarobi_v8][BallDetector] Ball candidate contours", ball_candidate_contours);
    cv::imshow("[alfarobi_v8][BallDetector] Input as HSV", input_hsv);

    auto t2 = boost::chrono::high_resolution_clock::now();

    auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();        
    std::cout << "[alfarobi_v8][BallDetector] Elapsed time: "<< (default_scalar_t)elapsed_time << " ms" << std::endl;
}

}