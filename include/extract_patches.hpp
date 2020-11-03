#include <math.h>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
//#include <torch/script.h>

void convertKeypoints(std::vector<cv::KeyPoint> &kpts,
                      std::vector<cv::Mat> &M,
                      std::vector<size_t> &out_pyr_idx,
                      int PS = 32,
                      float mag_factor = 12.0){
    M.clear();
    out_pyr_idx.clear();
    const double half_PS = double(PS) / 2.0;
    for (size_t i=0; i<kpts.size(); i++){
        cv::KeyPoint kp = kpts[i];
        cv::Mat M_cur(2, 3, CV_64F);
        const double s = kp.size * mag_factor / float(PS);
        const double pyr_idx = std::max(0.0, std::floor(std::log2(s)));
        const double d_factor = std::pow(2.0, pyr_idx);
        const double s_pyr = s / d_factor;
        const double sin_a = std::sin(kp.angle * M_PI / 180.0);
        const double cos_a = std::cos(kp.angle * M_PI / 180.0);
        M_cur.at<double>(0,0) = s_pyr*cos_a;
        M_cur.at<double>(0,1) = -s_pyr*sin_a;
        M_cur.at<double>(0,2) = (-s_pyr*cos_a + s_pyr*sin_a) * half_PS + kp.pt.x / d_factor;

        M_cur.at<double>(1,0) = s_pyr*sin_a;
        M_cur.at<double>(1,1) = s_pyr*cos_a;
        M_cur.at<double>(1,2) = (-s_pyr*sin_a - s_pyr*cos_a) * half_PS + kp.pt.y / d_factor;
        M.push_back(M_cur);
        out_pyr_idx.push_back(int(pyr_idx));
    }
}

std::vector<cv::Mat> build_image_pyramid(cv::Mat &img, int min_size = 16){
    std::vector<cv::Mat> out;
    cv::Mat cur_img = img.clone();
    out.push_back(img.clone());
    while (std::min(cur_img.cols, cur_img.rows) > min_size) {
        cv::pyrDown(cur_img.clone(), cur_img);
        out.push_back(cur_img);
    }
    return out;
}


void extract_patches(cv::Mat &img,
                     std::vector<cv::KeyPoint> &kpts,
                     std::vector<cv::Mat> &patches,
                     int PS = 32,
                     float mag_factor = 12.0){
    std::vector<cv::Mat> img_pyr = build_image_pyramid(img, int(float(PS)/2));
    size_t max_pyr_idx = img_pyr.size() - 1;

    std::vector<cv::Mat> M;
    std::vector<size_t> pyr_idxs;
    convertKeypoints(kpts, M, pyr_idxs, PS, mag_factor);
    patches.clear();
    patches.resize(kpts.size());
#ifdef _OPENMP
    omp_set_nested(1);
#endif
#pragma omp parallel for schedule (dynamic,1)
    for (size_t i=0; i<M.size(); i++){
        cv::Mat current_patch(PS,PS, CV_32F);
        cv::warpAffine(img_pyr[std::min(max_pyr_idx, pyr_idxs[i])],
                current_patch,
                M[i], cv::Size(PS,PS),
                cv::INTER_LINEAR + cv::WARP_INVERSE_MAP + cv::WARP_FILL_OUTLIERS,
                cv::BORDER_REPLICATE);
        patches[i] = current_patch;
    }
}

/*
void extract_tensor_patches(cv::Mat &img,
                     std::vector<cv::KeyPoint> &kpts,
                     torch::Tensor& patches,
                     int PS = 32,
                     float mag_factor = 10.0){
    cv::Mat grayImg;
    if (img.channels() > 1){
        cv::cvtColor(img, grayImg, cv::COLOR_BGR2GRAY);
    }
    else {
        grayImg = img;
    }
    std::vector<cv::Mat> img_pyr = build_image_pyramid(grayImg, int(float(PS)/2));
    size_t max_pyr_idx = img_pyr.size() - 1;
    std::vector<torch::Tensor> patches_list;
    std::vector<cv::Mat> M;
    std::vector<size_t> pyr_idxs;
    convertKeypoints(kpts, M, pyr_idxs, PS, mag_factor);
    for (size_t i=0; i<M.size(); i++){
        cv::Mat current_patch(PS,PS, CV_32F);
        cv::warpAffine(img_pyr[std::min(max_pyr_idx, pyr_idxs[i])],
                current_patch,
                M[i], cv::Size(PS,PS),
                cv::INTER_LINEAR + cv::WARP_INVERSE_MAP + cv::WARP_FILL_OUTLIERS,
                cv::BORDER_REPLICATE);
        torch::Tensor tensor_patch = torch::zeros({ PS, PS, 1 });
                  memcpy(tensor_patch.data_ptr(), current_patch.data, tensor_patch.numel() * sizeof(float));
                  tensor_patch = tensor_patch.permute({2,0,1});
                  patches_list.push_back(tensor_patch);
    }
    patches = torch::stack(torch::TensorList(patches_list));
}
*/
