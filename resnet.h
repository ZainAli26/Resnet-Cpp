#ifndef RESNET_H
#define RESNET_H

#include <torch/torch.h>
#include <torch/script.h>
#include <opencv2/opencv.hpp> 
#include <opencv2/imgproc/imgproc.hpp>
#include <memory>

class resnet
{
private:
    char* model_path;
    std::shared_ptr<torch::jit::script::Module> module;
    int input_image_size;
public:
    resnet(char* model_path="../../resnet.pt",int input_image_size=224);
    ~resnet();
    torch::Tensor preprocess(cv::Mat image);
    std::vector<int> predict(std::vector<cv::Mat>);
};
#endif
