#pragma once

#include <opencv2/core/core.hpp>

namespace alfarobi_v8{

class LBP{
private:
    auto lbp_ri(uchar binary){
        uchar min_binary(binary);
        uchar temp(binary);
        for(int i(0); i < 8; i++){        
            uchar end_bit = (0b1 & temp) << 7;
            temp = (temp >> 1) | end_bit;        
            if(temp < min_binary){
                min_binary = temp;
            }
        }
        return min_binary;
    }

    auto lbp_core(const cv::Mat& input, std::vector<double> &grid_hist, cv::Rect grid, cv::Mat& visual){
        int rows(input.rows);
        int cols(input.cols);
        const uchar* input_data = input.data;

        uchar *visual_data = visual.data;

        for(int i(grid.tl().y); i < grid.br().y; i++){
            int y1_limit_cond = i > 0 ? -1 : 0;
            int y2_limit_cond = i < rows ? -1 : 0;
            for(int j(grid.tl().x); j < grid.br().x; j++){
                int idx = i*cols + j;
                uchar thresh = input_data[idx];
                int kondisi_batas_x1 = j > 0 ? -1 : 0;
                int kondisi_batas_x2 = j < cols ? -1 : 0;
                int lbp = (((input_data[(idx+1) & kondisi_batas_x2] >= thresh) & kondisi_batas_x2) << 7 ) | (((input_data[(idx+cols+1) & y2_limit_cond&kondisi_batas_x2] >= thresh) & y2_limit_cond&kondisi_batas_x2) << 6) |
                            (((input_data[(idx+cols) & y2_limit_cond] >= thresh) & y2_limit_cond) << 5) | (((input_data[(idx+cols-1) & y2_limit_cond&kondisi_batas_x1] >= thresh) & y2_limit_cond&kondisi_batas_x1) << 4) |
                            (((input_data[(idx-1) & kondisi_batas_x1] >= thresh) & kondisi_batas_x1) << 3) | (((input_data[(idx-cols-1) & y1_limit_cond&kondisi_batas_x1] >= thresh) & y1_limit_cond&kondisi_batas_x1) << 2) |
                            (((input_data[(idx-cols) & y1_limit_cond] >= thresh) & y1_limit_cond) << 1) | (((input_data[(idx-cols+1) & y1_limit_cond&kondisi_batas_x2] >= thresh) & y1_limit_cond&kondisi_batas_x2));
                int lbpri = lbp;//(int)LBPri(lbp);
                grid_hist[lbpri]++;
                visual_data[idx] = lbpri;
            }
        }    
    }

public:
    auto extract(const cv::Mat& input, std::vector<double>& desc, cv::Mat& visual, int grid_size){
        int rows = input.rows;
        int cols = input.cols;

        while(1){
            static int pos_x = 0;
            static int pos_y = 0;
            std::vector<double> vtemp(256);
            cv::Rect sub_grid(pos_x, pos_y, grid_size, grid_size);        
            lbp_core(input, vtemp, sub_grid, visual);
            //-- normalizing by considering it as if a histogram        
            auto max_val(0.);
            for(auto it(vtemp.begin()); it != vtemp.end(); it++){
                if(*it > max_val){
                    max_val = *it;
                }
            }

            for(auto it(vtemp.begin()); it != vtemp.end(); it++){
                *it /= max_val;
            }

            desc.insert(desc.end(), vtemp.begin(), vtemp.end());
            pos_x += grid_size;
            if(pos_x >= cols){
                pos_x = 0;
                pos_y += grid_size;
                if(pos_y >= rows){
                    pos_x = 0;
                    pos_y = 0;
                    break;
                }
            }
        }
    }
};

}