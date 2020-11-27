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
    auto calcVariance(const cv::Mat& in, cv::Point p);
    auto getHist(const cv::Mat& ref, cv::MatND hist[3]);

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
    cv::MatND hist_reference_[3];

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
    cv::Mat ref( cv::imread("../data/ball_reference.jpeg") );
    getHist(ref, hist_reference_);
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
auto BallDetector<Search, Descriptor, Classifier, width, height>::calcVariance(const cv::Mat& in, cv::Point p){
    auto mean(.0);            
    auto radius(5);
    auto sz = pow(((2*radius)+1),2)-1;
    auto var(.0);
    for(int i(-radius*in.cols); i <= radius*in.cols; i += in.cols){
        for(int j(-radius); j <= radius; j++){
            int idx = p.y*in.cols + p.x + i + j;
            mean += in.at<float>(idx);
        }
    }
    mean /= sz; //-- sampling n-1
    for(int i(-radius*in.cols); i <= radius*in.cols; i += in.cols){
        for(int j(-radius); j <= radius; j++){
            int idx = p.y*in.cols + p.x + i + j;
            int temp = in.at<float>(idx) - mean;
            var += temp*temp;
        }
    }
    var /= sz; //-- sampling n-1
    return var;
}

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width, std::size_t height>
auto BallDetector<Search, Descriptor, Classifier, width, height>::getHist(const cv::Mat& ref, cv::MatND hist[3]){
    
    cv::Mat ball_mask( cv::Mat::zeros(ref.size(), CV_8UC1) );
    cv::Point center(ball_mask.cols/2, ball_mask.rows/2);
    cv::circle(ball_mask, center,
        (center.y < center.x)?center.y:center.x, cv::Scalar(1), cv::FILLED);
    
    int hist_bin[1] = {32};
    float ranges_hsv[3][2] = {{0., 180.}, {0., 256.}, {0., 256.}};
    for(int i(0); i < 3; i++){
        int chn[1] = {i};
        const float* range[1] = {ranges_hsv[i]};
        cv::calcHist(&ref, 1, chn, ball_mask,
                        hist[i], 1, hist_bin, range, true, false);
        cv::normalize(hist[i], hist[i], 0, 1, cv::NORM_MINMAX);
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
    cv::Mat output(_in.clone());

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
    
    contours.clear();
    std::vector<cv::Vec4i> hierarchy;
    cv::findContours(cropped_invert_field, contours, hierarchy,
                        cv::RETR_CCOMP, cv::CHAIN_APPROX_NONE);    
    std::vector<default_scalar_t> contour_area;
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

    auto [max_val, max_point, dt] = search_->search(ball_candidate_contours);

    //-- Threshold
    auto min_white_percentage(.2);
    auto max_white_percentage(.75);
    auto min_var(1.2); 
    auto score_limit(10.);
    //-- Optimal
    auto opt_score(.0);
    //--Ball index
    auto idx(-1);
    //-- print
    auto wht_percent_cout(.0);
    auto var_cout(.0);
    auto score_cout(.0);
    cv::Rect ball_rect;

    if(max_val.size())
        std::cout << "Candidates:\n  [Pos.] [Rad.] [Wht. Percent.] [Var.] [Hist. Score]" << std::endl;
    cv::Point ball_pos;
    for(std::size_t i(0); i < max_val.size(); i++){    	

        int pos_x = max_point[i].x - (int)max_val[i];
        int pos_y = max_point[i].y - (int)max_val[i];

        if(pos_x < 0) pos_x = 0;
        if(pos_y < 0) pos_y = 0;

        int d1 = 2. * max_val[i];
        int d2 = d1;

        if(pos_x + d1 >= width) d1 = width - pos_x;
        if(pos_y + d2 >= height) d2 = height - pos_y;

        cv::Mat roi(ball_color, cv::Rect(pos_x, pos_y, d1, d2));         
        
        auto variance(.0);
        auto score_hist(.0);

        auto white_percentage = (default_scalar_t)cv::countNonZero(roi)/(d1*d2);                        
        
        //=============== Decision Tree ==================
        if(white_percentage >= min_white_percentage
            && white_percentage <= max_white_percentage){            
            variance = calcVariance(dt, max_point[i]);
            if(variance > min_var){
                auto pass(true);
                if(!(hist_reference_[0].empty()
                    || hist_reference_[1].empty()
                    || hist_reference_[2].empty())){
                    cv::Mat roi2 = cv::Mat(input_hsv, cv::Rect(pos_x, pos_y, d1, d2));

                    cv::MatND hist_target[3];
                    getHist(roi2, hist_target);
                    auto min_score(std::numeric_limits<default_scalar_t>::max());
                    for(int j(0); j < 3; j++){
                        auto temp_score( cv::compareHist(hist_reference_[j], hist_target[j], 2) );
                        if(temp_score < min_score) min_score = temp_score;
                        if(temp_score < score_limit){   
                            pass = false;                         
                            break;
                        }
                    }
                    score_hist = min_score;
                }else{
                    pass = false;
                }

                if(pass){                    
                    //-- add some scale to get a bit more bigger
                    auto scale = (max_val[i] < 50.) ? 1.5 : 1.2;
                    int pos_x_2 = max_point[i].x - (int)(max_val[i]*scale);
                    int pos_y_2 = max_point[i].y - (int)(max_val[i]*scale);

                    pos_y_2 -= max_val[i] * .3;

                    if(pos_x_2 < 0) pos_x_2 = 0;
                    if(pos_y_2 < 0) pos_y_2 = 0;

                    int d1_2 = 2 * max_val[i] * scale;
                    int d2_2 = d1_2;

                    if(pos_x_2 + d1_2 >= ball_color.cols) d1_2 = ball_color.cols - pos_x_2;
                    if(pos_y_2 + d2_2 >= ball_color.rows) d2_2 = ball_color.rows - pos_y_2;

                    int d = std::min(d1_2, d2_2);
                    
                    cv::Mat roi_h(field_color, cv::Rect(pos_x_2, pos_y_2, d, d));
                    auto green_weight( countNonZero(roi_h) );
                    if(green_weight > .0){
	                    cv::Mat roi_frame(_in, cv::Rect(pos_x_2, pos_y_2, d, d));
	                    cv::Mat roi_gray(roi_frame.clone());
	                    cv::cvtColor(roi_gray, roi_gray, cv::COLOR_BGR2GRAY);               
	                                
	                    std::vector<default_scalar_t> desc;

                        cv::resize(roi_gray, roi_gray, cv::Size(64, 64));                        
                        cv::Mat dummy(cv::Mat::zeros(roi_gray.size(), CV_8UC1));

                        desc_->extract(roi_gray, desc, dummy, 16);

	                    cv::Mat mat_desc(cv::Mat(1, (int)desc.size(), CV_32F, desc.data()));

	                    auto label = classifier_->predict(mat_desc);

	                    if(label > 0.0){                    	

	                    	ball_rect = cv::Rect(pos_x_2, pos_y_2, d, d);  

	                        idx = i;

	                        opt_score = score_hist;                        

	                        var_cout = variance;
	                        score_cout = score_hist;
	                        wht_percent_cout = white_percentage;                                      
	                        	                        
                            cv::rectangle(output, ball_rect, cv::Scalar(255, 0, 255), 2);
	                    }  
	                }                                                     
                }            
            }             
        }
        //================= End of Decision Tree ===================

        std::cout << i+1 <<"." << max_point[i]
                         << " ; " << max_val[i]
                         << " ; " << white_percentage
                         << " ; " << variance
                         << " ; " << score_hist << std::endl;
    }    

    if(idx != -1){
	    ball_pos = max_point[idx];
	    // last_pos = pos;
	    cv::rectangle(output, ball_rect, cv::Scalar(0, 255, 0));
	    // circle(frame,max_point[idx],(int)max_val[idx],Scalar(0,255,0),2);
	}

    cv::imshow("[alfarobi_v8][BallDetector] Ball candidate contours", ball_candidate_contours);
    cv::imshow("[alfarobi_v8][BallDetector] Input as HSV", input_hsv);
    cv::imshow("[alfarobi_v8][BallDetector] Output", output);

    auto t2 = boost::chrono::high_resolution_clock::now();

    auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();        
    std::cout << "[alfarobi_v8][BallDetector] Elapsed time: "<< (default_scalar_t)elapsed_time << " ms" << std::endl;
}

}