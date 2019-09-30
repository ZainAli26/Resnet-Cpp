#include "resnet.h"

resnet::resnet(char* model_path, int input_image_size)
{
    this->model_path = model_path;
    this->input_image_size = input_image_size;
    this->module = torch::jit::load(model_path);
}

resnet::~resnet()
{
}

torch::Tensor resnet::preprocess(cv::Mat origin_image)
{
    cv::Mat resized_image;
    cv::cvtColor(origin_image, resized_image, cv::COLOR_RGB2BGR);
    cv::resize(resized_image, resized_image, cv::Size(input_image_size, input_image_size));
    cv::Mat img_float;
    resized_image.convertTo(img_float, CV_32F, 1.0 / 255);
    auto img_tensor = torch::CPU(torch::kFloat32).tensorFromBlob(img_float.data, {1, input_image_size, input_image_size, 3});
    img_tensor = img_tensor.permute({0, 3, 1, 2});
    img_tensor[0][0] = img_tensor[0][0].sub(0.485).div(0.229);
    img_tensor[0][1] = img_tensor[0][1].sub(0.456).div(0.224);
    img_tensor[0][2] = img_tensor[0][2].sub(0.406).div(0.225);
    auto img_var = torch::autograd::make_variable(img_tensor, false);
    return img_var;
}

std::vector<int> resnet::predict(std::vector<cv::Mat> images)
{
    std::vector<int> output_labels;
    std::vector<at::Tensor> inputs_vec;
    for(int i=0; i<images.size(); i++){
        inputs_vec.push_back(preprocess(images[i]));
    }
    at::Tensor input_ = torch::cat(inputs_vec);
    std::vector<torch::jit::IValue> inputs;
    inputs.push_back(input_.to(at::kCUDA));
    torch::Tensor output;
    output = module->forward(inputs).toTensor();
    for(int i =0; i<output.size(0); i++){
        at::Tensor max_ind = at::argmax(output[i]);
        output_labels.push_back(max_ind.item<int>());
        //std::cout << output[i] << std::endl;
    }
    return output_labels;
}