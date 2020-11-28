#pragma once

#include <iostream>

#include <boost/chrono.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

#include <fitting/fitcircle.h>

namespace alfarobi_v8{

using default_scalar_t = double;
constexpr std::size_t default_width{640};
constexpr std::size_t default_height{480};

template <typename Search, typename Descriptor, typename Classifier,
            std::size_t width = default_width, std::size_t height = default_height>
class BallDetector{
public:

    using Points = std::vector<cv::Point>;

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
    auto filterContourData(std::vector<cv::Mat> &divided_roi, cv::Point top_left_pt, std::vector<Points> &selected_data, cv::Mat *debug_mat, int sub_mode = 0);

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
    static constexpr auto FIT_CIRCLE_MAX_STEPS{20};
    static constexpr auto FIT_CIRCLE_EPS{1e-12};
    static constexpr auto MAX_FITTING_COST{8.};
    static constexpr auto MAX_BALL_RADIUS{width >> 1};
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
    auto sz = std::pow(((2*radius)+1),2)-1;
    auto var(.0);
    auto in_cpy(in.clone());
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
            int temp = in.at<default_scalar_t>(idx) - mean;
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
auto BallDetector<Search, Descriptor, Classifier, width, height>::filterContourData(std::vector<cv::Mat> &divided_roi, cv::Point top_left_pt,
                       std::vector<Points> &selected_data, cv::Mat *debug_mat, int sub_mode){
    int num_roi_cols = divided_roi.front().cols;
    int num_roi_rows = divided_roi.front().rows;
    bool horizon_scan = (float)num_roi_rows/(float)num_roi_cols < .75f;
    cv::Point map_origin[4];
    map_origin[0].x = top_left_pt.x;
    map_origin[0].y = top_left_pt.y;
    map_origin[1].x = (sub_mode == 2)?top_left_pt.x:top_left_pt.x + divided_roi.front().cols;
    map_origin[1].y = (sub_mode == 2)?top_left_pt.y + divided_roi.front().rows:top_left_pt.y;
    map_origin[2].x = top_left_pt.x;
    map_origin[2].y = top_left_pt.y + num_roi_rows;
    map_origin[3].x = top_left_pt.x + num_roi_cols;
    map_origin[3].y = top_left_pt.y + num_roi_rows;
    for(size_t idx = 0; idx < divided_roi.size(); idx++){

        int scan_mode = idx;

        switch(idx){
        case 0:scan_mode = (sub_mode == 1) ? 0 : (sub_mode == 2) ? 2 : horizon_scan ? 0 : 2;break;
        case 1:scan_mode = (sub_mode == 1) ? 1 : (sub_mode == 2) ? 3 : horizon_scan ? 1 : 2;break;
        case 2:scan_mode = horizon_scan ? 0 : 3;break;
        case 3:scan_mode = horizon_scan ? 1 : 3;break;
        }

        switch(scan_mode){
        case 0:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=0;j<num_roi_cols;j++){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 1:{
            for(int i=0;i<num_roi_rows;i++){
                for(int j=num_roi_cols-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(i,j) == 255){
                        if(j==num_roi_cols-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + j;
                        selected_point.y = map_origin[idx].y + i;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(i,j) = 255;
                        break;
                    }
                }
            }
        }break;
        case 2:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=0;j<num_roi_rows;j++){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==0)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;
        case 3:{
            for(int i=0;i<num_roi_cols;i++){
                for(int j=num_roi_rows-1;j>=0;j--){
                    if(divided_roi[idx].at<uchar>(j,i) == 255){
                        if(j==num_roi_rows-1)continue;
                        cv::Point selected_point;
                        selected_point.x = map_origin[idx].x + i;
                        selected_point.y = map_origin[idx].y + j;
                        selected_data[idx].push_back(selected_point);
                        debug_mat[idx].at<uchar>(j,i) = 255;
                        break;
                    }
                }
            }
        }break;

        }
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

    //-- threshold
    auto min_white_percentage(.3);
    auto max_white_percentage(.85);
    auto min_var(30.); 
    auto score_limit(.75);
    //-- optimal
    auto opt_score(.0);
    //-- ball index
    auto idx(-1);

    cv::Rect ball_rect;

    std::cout << "Candidate size : " << max_val.size() << std::endl;

    if(max_val.size())
        std::cout << "Candidates:\n  [Pos.] [Rad.] [Wht. Percent.] [Var.] [Hist. Score]" << std::endl;

    cv::Point ball_pos;
    for(std::size_t i(0); i < max_val.size(); i++){    	
        //-- add some scale to get a bit more bigger
        auto scale = (max_val[i] < 50.) ? 1.2 : 1.;
        int pos_x = max_point[i].x - (int)(max_val[i]*scale);
        int pos_y = max_point[i].y - (int)(max_val[i]*scale);

        if(pos_x < 0) pos_x = 0;
        if(pos_y < 0) pos_y = 0;

        int d1 = 2. * max_val[i] * scale;
        int d2 = d1;

        if(pos_x + d1 >= width) d1 = width - pos_x;
        if(pos_y + d2 >= height) d2 = height - pos_y;

        cv::Rect roi_rect(pos_x, pos_y, d1, d2);
        cv::Mat roi(ball_color, roi_rect);         
        
        auto variance(.0);
        auto score_hist(.0);

        auto white_percentage = (default_scalar_t)cv::countNonZero(roi)/(d1*d2);                        
        cv::Vec4f circle_param(-1.0f, -1.0f , std::numeric_limits<float>::max(), std::numeric_limits<float>::max());
        auto opt_diff_to_weighting(std::numeric_limits<default_scalar_t>::max());
        //-- Decision Tree
        if(white_percentage >= min_white_percentage
            && white_percentage <= max_white_percentage){            
            variance = calcVariance(dt, max_point[i]);
            if(variance > min_var){
                std::vector<cv::Mat> sub_frame1(2);
                std::vector<cv::Mat> sub_frame2(2);
                int sub_mode=0;
                sub_frame1[0] = cv::Mat(roi, cv::Rect(0, 0, roi_rect.width >> 1, roi_rect.height));
                sub_frame1[1] = cv::Mat(roi, cv::Rect(roi_rect.width >> 1, 0, roi_rect.width >> 1, roi_rect.height));
                //    cv::line(output_view,cv::Point(tl_pt.x+roi_rect.width/2,tl_pt.y),cv::Point(tl_pt.x+roi_rect.width/2,tl_pt.y+roi_rect.height),cv::Scalar(255,0,0),2);
                cv::Mat sub_sample1[2];
                sub_sample1[0] = cv::Mat::zeros(sub_frame1[0].size(), CV_8UC1);
                sub_sample1[1] = cv::Mat::zeros(sub_frame1[1].size(), CV_8UC1);                
                std::vector<Points > selected_data1(2);
                filterContourData(sub_frame1, roi_rect.tl(), selected_data1, sub_sample1, 1);

                sub_frame2[0] = cv::Mat(roi, cv::Rect(0, 0, roi_rect.width, roi_rect.height >> 1));
                sub_frame2[1] = cv::Mat(roi, cv::Rect(0, roi_rect.height >> 1, roi_rect.width, roi_rect.height >> 1));
                //    cv::line(output_view,cv::Point(tl_pt.x,tl_pt.y+roi_rect.height/2),cv::Point(tl_pt.x+roi_rect.width,tl_pt.y+roi_rect.height/2),cv::Scalar(255,0,0),2);
                cv::Mat sub_sample2[2];
                sub_sample2[0] = cv::Mat::zeros(sub_frame1[0].size(), CV_8UC1);
                sub_sample2[1] = cv::Mat::zeros(sub_frame1[1].size(), CV_8UC1);
                std::vector<Points > selected_data2(2);
                filterContourData(sub_frame2, roi_rect.tl(), selected_data2, sub_sample2, 2);

                for(size_t j(0); j < selected_data1.size(); j++){
                    cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data1[j], FIT_CIRCLE_MAX_STEPS, FIT_CIRCLE_EPS);
                    // if(sub_circle_param[2] > 0 && sub_circle_param[2] < width)
                    //     cv::circle(output, cv::Point(sub_circle_param[0], sub_circle_param[1]), sub_circle_param[2], cv::Scalar(0, 0, 255), 2);
                    auto diff_to_weighting = std::abs(max_val[i] - sub_circle_param[2]);
                    // std::cout << "Cost : " << sub_circle_param[3] << std::endl;
                    if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < MAX_FITTING_COST
                            && sub_circle_param[2] > std::max(roi.cols, roi.rows) >> 2
                            && sub_circle_param[2] < MAX_BALL_RADIUS
                            && diff_to_weighting < opt_diff_to_weighting){
                        circle_param = sub_circle_param;
                        opt_diff_to_weighting = diff_to_weighting;
                    }
                        
                }

                for(size_t j(0); j < selected_data2.size(); j++){
                    cv::Vec4f sub_circle_param = FitCircle::getInstance()->newtonPrattMethod(selected_data2[j], FIT_CIRCLE_MAX_STEPS, FIT_CIRCLE_EPS);
                    // if(sub_circle_param[2] > 0 && sub_circle_param[2] < width)
                    //     cv::circle(output, cv::Point(sub_circle_param[0], sub_circle_param[1]), sub_circle_param[2], cv::Scalar(0, 0, 255), 2);
                    auto diff_to_weighting = std::abs(max_val[i] - sub_circle_param[2]);
                    // std::cout << "Cost : " << sub_circle_param[3] << std::endl;
                    if(sub_circle_param[3] < circle_param[3] && sub_circle_param[3] < MAX_FITTING_COST
                            && sub_circle_param[2] > std::max(roi.cols, roi.rows) >> 2
                            && sub_circle_param[2] < MAX_BALL_RADIUS
                            && diff_to_weighting < opt_diff_to_weighting){
                        circle_param = sub_circle_param;
                        opt_diff_to_weighting = diff_to_weighting;
                    }                        
                }                                  
                //--

                max_point[i] = cv::Point(circle_param[0], circle_param[1]);
                if(max_point[i].x < 0 & max_point[i].y < 0)continue;

                if(circle_param[2] > 0 && circle_param[2] < width){
                    cv::circle(output, cv::Point(circle_param[0], circle_param[1]), circle_param[2], cv::Scalar(255, 0, 0), 2);
                    cv::putText(output, std::to_string(i+1), max_point[i], cv::FONT_HERSHEY_SIMPLEX,0.5, cv::Scalar(0, 0, 255), 1);
                }

                pos_x = max_point[i].x - circle_param[2];
                pos_y = max_point[i].y - circle_param[2];

                if(pos_x < 0) pos_x = 0;
                if(pos_y < 0) pos_y = 0;

                d1 = 2. * circle_param[2];
                d2 = d1;

                if(pos_x + d1 >= width) d1 = width - pos_x;
                if(pos_y + d2 >= height) d2 = height - pos_y;

                roi_rect = cv::Rect(pos_x, pos_y, d1, d2);              
                roi = cv::Mat(ball_color, roi_rect);                
                //--       

                auto pass(true);
                if(!(hist_reference_[0].empty()
                    || hist_reference_[1].empty()
                    || hist_reference_[2].empty())){
                    cv::Mat hsv_roi = cv::Mat(input_hsv, roi_rect);

                    cv::MatND hist_target[3];
                    getHist(hsv_roi, hist_target);
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
                    auto scale(1.1);
                    int pos_x_2 = max_point[i].x - (int)(circle_param[2]*scale);
                    int pos_y_2 = max_point[i].y - (int)(circle_param[2]*scale);

                    // pos_y_2 -= circle_param[2] * .3;

                    if(pos_x_2 < 0) pos_x_2 = 0;
                    if(pos_y_2 < 0) pos_y_2 = 0;

                    int d1_2 = 2 * circle_param[2] * scale;
                    int d2_2 = d1_2;

                    if(pos_x_2 + d1_2 >= ball_color.cols) d1_2 = ball_color.cols - pos_x_2;
                    if(pos_y_2 + d2_2 >= ball_color.rows) d2_2 = ball_color.rows - pos_y_2;

                    int d = std::min(d1_2, d2_2);
                    cv::Mat roi_h(field_color, cv::Rect(pos_x_2, pos_y_2, d, d));
                    auto green_weight( cv::countNonZero(roi_h) );
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

                            max_val[i] = circle_param[2];
	                        	                        
                            cv::rectangle(output, ball_rect, cv::Scalar(255, 0, 255), 2);
	                    }  
	                }                                                     
                }            
            }             
        }
        //-- End of Decision Tree

        std::cout << i+1 <<"." << max_point[i]
                         << " ; " << max_val[i]
                         << " ; " << white_percentage
                         << " ; " << variance
                         << " ; " << score_hist << std::endl;
    }    

    if(idx != -1){
	    ball_pos = max_point[idx];
	    // last_pos = pos;
	    cv::circle(output, max_point[idx], (int)max_val[idx], cv::Scalar(0, 255, 0), 2);
	}

    cv::imshow("[alfarobi_v8][BallDetector] Ball candidate contours", ball_candidate_contours);
    // cv::imshow("[alfarobi_v8][BallDetector] Input as HSV", input_hsv);
    cv::imshow("[alfarobi_v8][BallDetector] Output", output);

    auto t2 = boost::chrono::high_resolution_clock::now();

    auto elapsed_time = boost::chrono::duration_cast<boost::chrono::milliseconds>(t2-t1).count();        
    std::cout << "[alfarobi_v8][BallDetector] Elapsed time: "<< (default_scalar_t)elapsed_time << " ms" << std::endl;
}

}