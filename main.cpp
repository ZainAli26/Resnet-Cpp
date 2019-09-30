#include<iostream>
#include <opencv2/opencv.hpp>
#include "resnet.h"
#include <sstream>
#include <string>
#include<fstream>

using namespace std;

int main(int argc, const char* argv[]){
    string model_path = argv[1];
    cout << "Here-1" << endl;
    cv::Mat origin_image = cv::imread(argv[2], CV_LOAD_IMAGE_COLOR);
    resnet obj = resnet(const_cast<char*>(model_path.c_str()));
    while(true){
        std::vector<cv::Mat> images;
        images.push_back(origin_image);
        std::vector<int> out = obj.predict(images);
        for(int i=0; i<out.size(); i++){
            cout << "Selected index: " << i << " " << out[i] << endl;
        }
    }
    cout << "Running Correctly" << endl;
    // string file_labels = "/home/smartcart/Desktop/Zain/Resnet/labels.txt";
    // string file_labels2 = "/home/smartcart/Desktop/Zain/Resnet/barcodes.txt";
    // ifstream infile;
    // string STRING;
    // infile.open (file_labels2);
    // int a=0;
    // string previousLine="";
    // while(getline(infile,STRING)) // To get you all the lines.
    // {
    //     //; // Saves the line in STRING.
    //     if (STRING != previousLine)
    //     {
    //         previousLine=STRING;
    //         std::string delimiter = ",";
    //         size_t pos = STRING.find(delimiter);
    //         STRING.erase(0, pos + delimiter.length());
    //         string token = STRING.substr(0, pos);
    //         cout<<token<<endl; // Prints our STRING.
    //     }
    //     // else{
    //     //     break;
    //     // }

    // }
    // infile.close();
}