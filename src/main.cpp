#include <iostream>
#include <string>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/features2d/features2d.hpp>


#include "extract_patches.hpp"



#ifdef _OPENMP
#include <omp.h>
#endif


int main(int argc, char **argv)
{
    cv::Mat image = cv::imread("kpi.png", 0);
    std::vector<cv::KeyPoint> keypoints1;
    cv::Ptr<cv::AKAZE> detector = cv::AKAZE::create();
    detector->detect(image, keypoints1);

    std::vector<cv::Mat>  patches;
    extract_patches(image,
                    keypoints1,
                    patches, 64, 12.0);

    for (int i=0; i<10; i++){
        cv::imwrite(std::to_string(i) + ".png", patches[i]);
    }
    return 0;
}
