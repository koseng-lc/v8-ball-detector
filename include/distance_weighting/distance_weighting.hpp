/**
*   @author koseng (Lintang)
*   @brief Approximation of Distance Transform algorithm in only one pass
*/

#pragma once

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

namespace alfarobi_v8{

class DistanceWeightingSearch{
public:
    using max_val_t = std::vector<double>;
    using max_point_t = std::vector<cv::Point>;
public:
    auto distanceWeighting(const cv::Mat &in){
        cv::Mat weighting(cv::Mat::zeros(in.size(), CV_64FC1));
        auto num_rows(in.rows);
        auto num_cols(in.cols);
        for(auto i(1); i < (num_rows - 1); i++){		
            const uchar *in_ptr1 = in.ptr<uchar>(i-1);
            const uchar *in_ptr2 = in.ptr<uchar>(i);
            const uchar *in_ptr3 = in.ptr<uchar>(i+1);
            double *weighting_ptr1 = weighting.ptr<double>(i-1);
            double *weighting_ptr2 = weighting.ptr<double>(i);
            double *weighting_ptr3 = weighting.ptr<double>(i+1);
            for(auto j(1); j < (num_cols - 1); j++){
                if(in_ptr2[j]){
                    if(in_ptr3[j-1] !=0 && in_ptr3[j] !=0 && in_ptr3[j+1] !=0 && 
                        in_ptr2[j-1] !=0 && in_ptr2[j] !=0 && in_ptr2[j+1] !=0 &&
                        in_ptr1[j-1] !=0 && in_ptr1[j] !=0 && in_ptr1[j+1] !=0){					
                        double min_dist = fminf(fminf(weighting_ptr2[j-1], weighting_ptr1[j-1]),
                                                fminf(weighting_ptr1[j], weighting_ptr1[j+1]));				
                        weighting_ptr2[j] = min_dist + 1;
                    }else{
                        weighting_ptr2[j] = 1;
                    }
                }
            }
        }

        return weighting;
    }

    auto search(const cv::Mat& in){
        cv::Mat dt( distanceWeighting(in) );
        std::vector<std::vector<cv::Point>> contours;
        cv:findContours(in, contours, cv::RETR_EXTERNAL, cv::CHAIN_APPROX_NONE);

        cv::normalize(dt,dt, 0., 1., cv::NORM_MINMAX); 
        cv::imshow("[alfarobi_v8][DistanceWeightingSearch] Distance Weighting", dt);
        max_val_t max_val;
        max_point_t max_point;
        
        for(std::size_t i(0); i < contours.size(); i++){
            auto area = cv::contourArea(contours[i]);
            cv::Rect r = cv::boundingRect(contours[i]);        
            if(area <= MIN_BALL_AREA) continue;
            if(r.area() < 400){ //-- different treatment
                cv::Mat interest(dt, r);            
                cv::Point minp, maxp, ctr;
                double minv, maxv;
                cv::minMaxLoc(interest, &minv, &maxv, &minp, &maxp);                       
                if(maxv <= MIN_BALL_RADIUS)continue;
                ctr = cv::Point(r.tl().x + maxp.x, r.tl().y + maxp.y);
                //-- remove visited candidate
                cv::circle(dt, ctr, maxv, cv::Scalar(0), cv::FILLED);
                max_point.push_back(ctr);
                max_val.push_back(maxv);
            }else {
                for(auto j(0); j < MAX_CANDIDATE; j++){
                    cv::Mat interest(dt, r);
                    cv::Point minp, maxp, ctr;
                    double minv, maxv;
                    cv::minMaxLoc(interest, &minv, &maxv, &minp, &maxp);                
                    if(maxv <= MIN_BALL_RADIUS)break;
                    ctr = cv::Point(r.tl().x + maxp.x, r.tl().y + maxp.y);
                    //-- remove visited candidate               
                    cv::circle(dt, ctr, maxv, cv::Scalar(0), cv::FILLED);
                    max_point.push_back(ctr);        		
                    max_val.push_back(maxv);
                    
                }
            }
        }               
        return std::make_tuple(max_val, max_point);
    }
private:
    static constexpr auto MIN_BALL_RADIUS{5.};
    static constexpr auto MIN_BALL_AREA{100.};
    static constexpr auto MAX_CANDIDATE{5};
};

}