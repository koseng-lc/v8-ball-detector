/**
*   @author koseng (Lintang)
*/

#include <boost/thread.hpp>

#include <opencv2/videoio/videoio.hpp>

#include <distance_weighting/distance_weighting.hpp>
#include <lbp/lbp.hpp>
#include <classifier_wrapper/classifier_wrapper.hpp>
#include <v8_ball_detector/ball_detector.hpp>

int main(int argc, char** argv){

    alfarobi_v8::DistanceWeightingSearch dws;
    alfarobi_v8::LBP lbp;
    alfarobi_v8::ClassifierWrapper cw("../data/bola_lbp_standar.xml");

    alfarobi_v8::BallDetector<
        alfarobi_v8::DistanceWeightingSearch,
        alfarobi_v8::LBP,
        alfarobi_v8::ClassifierWrapper> bd(&dws, &lbp, &cw);
    bd.loadConfig("../data/v8_ball_detector_config.yaml");

    cv::VideoCapture vc("../video_test/video9.avi");    

    cv::Mat input(alfarobi_v8::default_height,
                  alfarobi_v8::default_width, CV_8UC3);    
    while(1){
        vc >> input;
        cv::flip(input, input, -1);
        bd.execute(input);
        if(cv::waitKey(0) == 27)break;
        boost::this_thread::sleep_for(boost::chrono::milliseconds(10));
    }

    return .0;
}