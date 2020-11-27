/**
*   @author koseng (Lintang)
*   @brief color calibrator
*/

#include <iostream>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/videoio/videoio.hpp>

constexpr auto hue_range{180};
constexpr auto sat_range{256};
cv::Mat all_in_one;
cv::MatND hist_reference[3];
cv::Mat color_table(hue_range, sat_range, CV_8UC1);

struct BallParam{
    int min_val{0};
    int max_val{255};
    int min_sat{0};
    int max_sat{255};
}ball_param;

struct MouseStuff{
    bool left_down{false};
    bool left_up{false};

    cv::Point coor1;
    cv::Point coor2;
}mouse_stuff;

auto loadConfig(){
    cv::FileStorage fs("../data/v8_ball_detector_config.yaml", cv::FileStorage::READ);    
    fs["color_table"] >> color_table;
    fs["min_val"] >> ball_param.min_val;
    fs["max_val"] >> ball_param.max_val;
    fs["min_sat"] >> ball_param.min_sat;
    fs["max_sat"] >> ball_param.max_sat;
    fs.release();
}

auto setupTrackbar(){
    std::string param_winname("[alfarobi_v8][Calibration] Parameters config");
    cv::namedWindow(param_winname, cv::WINDOW_NORMAL);

    cv::createTrackbar("Ball Min. Val", param_winname, &ball_param.min_val, 255);
    cv::createTrackbar("Ball Max. Val", param_winname, &ball_param.max_val, 255);
    cv::createTrackbar("Ball Min. Sat", param_winname, &ball_param.min_sat, 255);
    cv::createTrackbar("Ball Max. Sat", param_winname, &ball_param.max_sat, 255);
}

auto modifyTable(const cv::Mat& data, int flag){
    cv::Mat temp;
    cv::FileStorage fs("../data/v8_ball_detector_config.yaml", cv::FileStorage::READ);    
    fs["color_table"] >> temp;
    fs.release();

    if(!temp.data)
        temp = cv::Mat::zeros(180, 256, CV_8UC1);

    for(int i(0); i < data.rows; i++){            
        const cv::Vec3b* data_ptr = data.ptr<cv::Vec3b>(i);
        for(int j(0); j < data.cols; j++){
            int hue = data_ptr[j][0];
            int sat = data_ptr[j][1];
            temp.at<uchar>(hue, sat) = flag;
            color_table.at<uchar>(hue, sat) = flag;
        }
    }

    fs.open("../data/v8_ball_detector_config.yaml", cv::FileStorage::WRITE);
    fs << "color_table" << temp;
    fs << "min_val" << ball_param.min_val;
    fs << "max_val" << ball_param.max_val;
    fs << "min_sat" << ball_param.min_sat;
    fs << "max_sat" << ball_param.max_sat;
    fs.release();    
}

void getHist(const cv::Mat& ref, cv::MatND hist[3]){
    
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

auto mouse_call(int event, int x, int y, int flags, void* data){
    if(event == cv::EVENT_LBUTTONDBLCLK & !mouse_stuff.left_down){
    	if(x >= 640 & y < 480){
        	mouse_stuff.left_down = true;

        	mouse_stuff.coor1.x = std::max(std::min(x, 1279), 640);
        	mouse_stuff.coor1.y = std::max(std::min(y, 479), 0);          
        	
        	std::cout << "[alfarobi_v8][Calibration] Corner 1: " << mouse_stuff.coor1 << std::endl;
        }else{
        	std::cout << "[alfarobi_v8][Calibration] Crop in HSV area only!" << std::endl;
        }

    }

    if(event == cv::EVENT_LBUTTONDBLCLK & mouse_stuff.left_down){    	
        if(std::abs(x-mouse_stuff.coor1.x) > 1
            & std::abs(y-mouse_stuff.coor1.y) > 1) {
            mouse_stuff.left_up = true;

            mouse_stuff.coor2.x = std::max(std::min(x, 1279), 640);
        	mouse_stuff.coor2.y = std::max(std::min(y, 479), 0);

            std::cout << "[alfarobi_v8][Calibration] Corner 2: " << mouse_stuff.coor2 << std::endl;
        }
        else{
            std::cout << "[alfarobi_v8][Calibration] Create a region more than 1x1!" << std::endl;
        }
    }

    if(mouse_stuff.left_down
        & !mouse_stuff.left_up){

        cv::Point pt;
        pt.x = std::max(std::min(x, 1279), 640);
        pt.y = std::max(std::min(y, 479), 0);

        cv::Mat temp( all_in_one.clone() );
        cv::rectangle(temp, mouse_stuff.coor1, pt, cv::Scalar(0,0,255));
        cv::imshow("[alfarobi_v8][Calibration] Calibrator", temp);
    }

    if(mouse_stuff.left_down
        & mouse_stuff.left_up){

        cv::Rect box;
        box.width = std::abs(mouse_stuff.coor1.x - mouse_stuff.coor2.x);
        box.height = std::abs(mouse_stuff.coor1.y - mouse_stuff.coor2.y);
        box.x = std::min(mouse_stuff.coor1.x, mouse_stuff.coor2.x);
        box.y = std::min(mouse_stuff.coor1.y, mouse_stuff.coor2.y);       

        cv::Mat sel_roi(all_in_one, box);
        cv::imshow("[alfarobi_v8][Calibration] Selected ROI", sel_roi);

        mouse_stuff.left_down = false;
        mouse_stuff.left_up = false;

        while(1){
            int key = cv::waitKey(5) & 0xFF;
            if(key == 'f'){
                std::cout << "[alfarobi_v8][Calibration] Field color tagged!" << std::endl;
                modifyTable(sel_roi, 1);
                break;
            }else if(key == 'b'){
                std::cout << "[alfarobi_v8][Calibration] Ball color tagged!." << std::endl;
                modifyTable(sel_roi, 2);
                break;
            }else if(key == 'd'){
                std::cout << "[alfarobi_v8][Calibration] Color tag deleted!" << std::endl;
                modifyTable(sel_roi, 0);
                break;
            }else if(key == 'h'){
                std::cout << "[alfarobi_v8][Calibration] Ball color histogram saved!"<< std::endl;
                cv::imwrite("../data/ball_reference.jpeg", sel_roi);
                break;
            }else if(key == 32)break;
        }
        cv::destroyWindow("[alfarobi_v8][Calibration] Selected ROI");
    }
}

auto segmentField(const cv::Mat& _in_hsv){
    cv::Mat field_color(cv::Mat::zeros(_in_hsv.size(), CV_8UC1));
    
    for(auto i(0); i < _in_hsv.total(); i++){
        int hue = _in_hsv.at<cv::Vec3b>(i)[0];
        int sat = _in_hsv.at<cv::Vec3b>(i)[1];                
        int val = _in_hsv.at<cv::Vec3b>(i)[2];

        if(color_table.at<uchar>(hue*sat_range + sat) == 1)
            field_color.at<uchar>(i) = 255;
    }

    return field_color;
}

auto segmentBall(const cv::Mat& _in_hsv){
    cv::Mat ball_color(cv::Mat::zeros(_in_hsv.size(), CV_8UC1));
    
    for(auto i(0); i < _in_hsv.total(); i++){
        int sat = _in_hsv.at<cv::Vec3b>(i)[1];                
        int val = _in_hsv.at<cv::Vec3b>(i)[2];

        if(val >= ball_param.min_val & sat >= ball_param.min_sat
            & val <= ball_param.max_val & sat <= ball_param.max_sat)
            ball_color.at<uchar>(i) = 255;
    }

    return ball_color;
}

int main(int argc, char** argv){
    loadConfig();
    setupTrackbar();
    cv::VideoCapture vc("../video_test/video9.avi");
    cv::Mat input;
    cv::Mat input_hsv;
    cv::setMouseCallback("[alfarobi_v8][Calibration] Calibrator", mouse_call, 0);
    while(1){
        vc >> input;
        cv::cvtColor(input, input_hsv, cv::COLOR_BGR2HSV);
        
        cv::hconcat(input, input_hsv, all_in_one);

        cv::Mat field_color( segmentField(input_hsv) );
        cv::Mat ball_color( segmentBall(input_hsv) );

        cv::Mat temp;
        cv::hconcat(field_color, ball_color, temp);
        cv::cvtColor(temp, temp, cv::COLOR_GRAY2BGR);

        cv::vconcat(all_in_one, temp, all_in_one);

        cv::putText(all_in_one, "RGB", cv::Point(5, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0,0,255), 2);
        cv::putText(all_in_one, "HSV", cv::Point(645, 15), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(255,0,0), 2);
        cv::putText(all_in_one, "FIELD", cv::Point(5, 495), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0,0,255), 2);
        cv::putText(all_in_one, "BALL", cv::Point(645, 495), cv::FONT_HERSHEY_SIMPLEX, .5, cv::Scalar(0,0,255), 2);

        cv::imshow("[alfarobi_v8][Calibration] Calibrator", all_in_one);                
        while(1){
            cv::setMouseCallback("[alfarobi_v8][Calibration] Calibrator", mouse_call, 0);
            int key = cv::waitKey(0) & 0xFF;
            if(key == 32)break;
        }
    }

    return 0.;
}