#include <iostream>
#include <queue>
#include <iterator>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <chrono>
#include "INIReader.h"

#include <opencv2/core.hpp>
#include <opencv2/dnn.hpp>
#include <opencv2/dnn/all_layers.hpp>
#include <opencv2/core/utils/filesystem.hpp>

#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>


//constexpr float CONFIDENCE_THRESHOLD = 0;
//constexpr float NMS_THRESHOLD = 0.4;
//constexpr int NUM_CLASSES = 80;

// colors for bounding boxes
const cv::Scalar colors[] = {
    {0, 255, 255},
    {255, 255, 0},
    {0, 255, 0},
    {255, 0, 0}
};
const auto NUM_COLORS = sizeof(colors) / sizeof(colors[0]);

void postprocess(cv::Mat frame, int NUM_CLASSES, int CONFIDENCE_THRESHOLD, int NMS_THRESHOLD, std::vector<cv::Mat> detections, std::vector<std::vector<int>>& indices, std::vector<std::vector<cv::Rect>>& boxes, std::vector<std::vector<float>>& scores)
{
    //detect
#if 0
    for (auto& output : detections)
    {
        const auto num_boxes = output.rows;
        //std::cout << num_boxes << std::endl;
        for (int i = 0; i < num_boxes; ++i)
        {
            auto x = output.at<float>(i, 0) * frame.cols;
            auto y = output.at<float>(i, 1) * frame.rows;
            auto width = output.at<float>(i, 2) * frame.cols;
            auto height = output.at<float>(i, 3) * frame.rows;
            cv::Rect rect(x - width / 2, y - height / 2, width, height);

            for (int c = 0; c < NUM_CLASSES; ++c)
            {
                auto confidence = *output.ptr<float>(i, 5 + c);
                if (confidence > CONFIDENCE_THRESHOLD)
                {
                    boxes[c].push_back(rect);
                    scores[c].push_back(confidence);
                }
            }
        }
    }
#else   
    for (size_t i = 0; i < detections.size(); ++i)
    {
        float* data = (float*)detections[i].data;
        for (size_t j = 0; j < detections[i].rows; ++j, data += detections[i].cols)
        {
            cv::Mat score = detections[i].row(j).colRange(5, detections[i].cols);
            cv::Point classIdPoint;
            double confidence;

            // Get the value and location of the maximum score
            minMaxLoc(score, 0, &confidence, 0, &classIdPoint);

            if (confidence > CONFIDENCE_THRESHOLD)
            {
                auto x = (float)(data[0] * frame.cols);
                auto y = (float)(data[1] * frame.rows);
                auto width = (float)(data[2] * frame.cols);
                auto height = (float)(data[1] * frame.rows);
                cv::Rect rect(x - width / 2, y - height / 2, width, height);
                int c = classIdPoint.x;
                boxes[c].push_back(rect);
                scores[c].push_back(confidence);
            }

        }
    }
#endif

    //non-maximum suppress
    for (int c = 0; c < NUM_CLASSES; ++c)
    {
        //std::cout << "Size before NMS: " << boxes[c].size() << std::endl;
        cv::dnn::NMSBoxes(boxes[c], scores[c], 0.0, NMS_THRESHOLD, indices[c]);
        //std::cout << "Size after NMS: " << boxes[c].size() << std::endl;
    }
}

cv::Rect Render(cv::Mat frame, int c, int i, std::vector<std::string> class_names,std::vector<std::vector<int>> indices, std::vector<std::vector<cv::Rect>> boxes, std::vector<std::vector<float>> scores, std::ofstream& log_file, int rWidth = 0, int rHeight = 0)
{
    const auto color = colors[c % NUM_COLORS];

    auto idx = indices[c][i];
    auto& rect = boxes[c][idx];
    rect.x += rWidth;
    rect.y += rHeight;
    if (rWidth != 0 && rHeight != 0)
        log_file << "Ping: " << rect.x << " " << rect.y << std::endl;
    std::ostringstream label_ss;
    label_ss << class_names[c] << ": " << std::fixed << std::setprecision(2) << scores[c][idx];

    auto label = label_ss.str();

    int baseline;
    auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

    cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

    if (rect.y - label_bg_sz.height - baseline < 10)
    {
        //label.c_str()
        cv::rectangle(frame, cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - label_bg_sz.height - baseline - 10), cv::Point(rect.x + rect.width, rect.y + rect.height), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    else
    {
        cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        cv::putText(frame, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
    }
    return boxes[c][idx];
}

int main()
{
    INIReader ini("config.ini");
    int NUM_CLASSES = 0;
    std::vector<std::string> class_names, class_names_LP;
    float CONFIDENCE_THRESHOLD = ini.GetReal("PARAMS", "confidence_threshold", 0);
    float NMS_THRESHOLD = ini.GetReal("PARAMS", "non_maximum_suppresion_threshold", 0.4);
    //{ 
    cv::String classPath = ini.GetString("PRIMARY", "class", "classes.txt"); //"model.names";
    std::ifstream class_file(classPath);
    if (!class_file)
    {
        std::cerr << "failed to open classes.txt\n";
        return 0;    
    }

    std::string line;
    while (std::getline(class_file, line))
    {
        class_names.push_back(line);
        ++NUM_CLASSES;
    }

    cv::String logPath = ini.GetString("FILE", "log", "logs.txt");
    std::ofstream log_file(logPath);
    cv::String cfgPath = ini.GetString("PRIMARY", "config", ""); //"model.cfg";
    cv::String weightPath = ini.GetString("PRIMARY", "weight", ""); //"model.weights";
    auto net = cv::dnn::readNetFromDarknet(cfgPath, weightPath);
    /*GPU mode*/
    net.setPreferableBackend(cv::dnn::DNN_BACKEND_CUDA);
    net.setPreferableTarget(cv::dnn::DNN_TARGET_CUDA);
    /*CPU mode*/
    //net.setPreferableBackend(cv::dnn::DNN_BACKEND_OPENCV);
    //net.setPreferableTarget(cv::dnn::DNN_TARGET_CPU);
    auto output_names = net.getUnconnectedOutLayersNames();
    
    cv::Mat frame, blob;
    std::vector<cv::Mat> detections;

    // Get all image in the folder
    std::vector<cv::String> filenames;
    cv::String imgPath = ini.GetString("FILE", "input", ""); //"//192.168.0.199/ubay_share/Container data/Truck_2/*.png
    cv::String outPath = ini.GetString("FILE", "output", "");
 
    std::vector<cv::String> extensions = {"JPG", "jpg", "jfif", "png"};//, "png"
    std::vector<cv::String> currentExtStr;
    for (int i = 0; i < extensions.size(); ++i)
    {
        cv::glob(imgPath + "/*." + extensions[i], currentExtStr);
        filenames.insert(filenames.end(), currentExtStr.begin(), currentExtStr.end());
    }
    if (cv::utils::fs::exists(outPath) == false)
        cv::utils::fs::createDirectory(outPath);

    /*Video source*/
    //cv::VideoCapture source(imgPath);

    size_t nFiles = filenames.size();
    log_file << "The number of images is " << nFiles << std::endl;
    std::cout << "The number of images is " << nFiles << std::endl;
    //cv::String ext; //image extension name
    cv::String basename; //image base name
    cv::String annFilePath; //annotation file name
    cv::String detection_result; //detection result
    cv::String fold;    //folder contains cropped region
    cv::String name;    //cropped region file name
    
    double dRows; // 1 per image number of rows
    double dCols; // 1 per image number of columns
    double x;
    double y;

    

    int k;
    for(size_t i = 0; i < nFiles; ++i)
    {
        //detection_result = "";
        log_file << "Image name: " << filenames[i] << std::endl;
        frame = cv::imread(filenames[i]);
        dRows = 1. / frame.rows;
        dCols = 1. / frame.cols;
        //ext = filenames[i].substr(filenames[i].find_last_of("."));// size() - 4);
        annFilePath = filenames[i].substr(0, filenames[i].find_last_of(".")) + ".txt";
        basename = annFilePath.substr(annFilePath.find_last_of("\\") + 1);
        basename = basename.substr(0, basename.find_last_of("."));
        std::cout << "annotation: " << annFilePath << std::endl;
        //std::cout << "basename: " << basename << std::endl;
        //if (i == 5)
        //    break;
        std::ofstream annoFile(annFilePath, std::ofstream::out);
        if (frame.empty())
        {
            cv::waitKey();
            break;
        }

        //auto total_start = std::chrono::steady_clock::now();
        cv::dnn::blobFromImage(frame, blob, 0.00392, cv::Size(608, 608), cv::Scalar(), true, false, CV_32F);
        net.setInput(blob);

        auto dnn_start = std::chrono::steady_clock::now();
        try
        {
            net.forward(detections, output_names);
        }
        catch (std::exception& ex)
        {
            log_file << "We got a problem!: " << ex.what();
            std::cout << "We got a problem!: " << ex.what();
            break;
        }
        auto dnn_end = std::chrono::steady_clock::now();

        std::vector<std::vector<int>> indices(NUM_CLASSES);
        std::vector<std::vector<cv::Rect>> boxes(NUM_CLASSES); //bounding boxes
        std::vector<std::vector<float>> scores(NUM_CLASSES); //confidence scores

        postprocess(frame, NUM_CLASSES, CONFIDENCE_THRESHOLD, NMS_THRESHOLD, detections, indices, boxes, scores);

        k = 0;
        int baseline;
        //label classes
        for (int c = 0; c < NUM_CLASSES; ++c)
        {
            for (size_t j = 0; j < indices[c].size(); ++j)
            {
                const auto color = colors[c % NUM_COLORS];

                auto idx = indices[c][j];
                auto& rect = boxes[c][idx];
                cv::Size s = frame.size();

                if (rect.x < 0)
                {
                    rect.width += rect.x;
                    rect.x = 0;
                }
                if (rect.x + rect.width >= s.width)
                    rect.width = s.width - rect.x;
                if (rect.y < 0)
                {
                    rect.height += rect.y;
                    rect.y = 0;
                }
                if (rect.y + rect.height >= s.height)
                    rect.height = s.height - rect.y;

                cv::Rect roi(rect.x, rect.y, rect.width, rect.height);
                cv::Range rows(rect.x, rect.x + rect.width);
                cv::Range cols(rect.y, rect.y + rect.height);
                cv::Mat matRoi = frame(cols, rows);

                try
                {
                    fold = outPath + "/" + class_names[c];
                    if (cv::utils::fs::exists(fold) == false)
                        cv::utils::fs::createDirectory(fold);
                    name = fold + "/" + basename + "_" + std::to_string(k) + ".png";
                    std::cout << "Cropped region name: " << name << std::endl;
                    cv::imwrite(name, matRoi);
                    ++k;
                    x = rect.x + rect.width / 2.0;// - 1;
                    y = rect.y + rect.height / 2.0;// - 1;
                    
                    annoFile << std::to_string(int(c)) << " " << x * dCols
                        << " " << y * dRows
                        << " " << rect.width * dCols
                        << " " << rect.height * dRows << "\n";
                }
                catch (const std::exception& ex)
                {
                    log_file << "Error while rendering ground truth image: " << ex.what() << std::endl;
                }

                log_file << class_names[c] << ":" << rect.x << " " << rect.y << std::endl;
        //        auto label_bg_sz = cv::getTextSize(class_names[c].c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);

        //        cv::rectangle(frame, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

        //        /*Label on primary detection result*/
        //        if (rect.y - label_bg_sz.height - baseline < 10)
        //        {
        //            //label.c_str()
        //            cv::rectangle(frame, cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - label_bg_sz.height - baseline - 10), cv::Point(rect.x + rect.width, rect.y + rect.height), color, cv::FILLED);
        //            cv::putText(frame, class_names[c].c_str(), cv::Point(rect.x + rect.width - label_bg_sz.width, rect.y + rect.height - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        //        }
        //        else
        //        {
        //            cv::rectangle(frame, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
        //            cv::putText(frame, class_names[c].c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
        //        }
            }
        }

        //
        //while (cv::waitKey(1) < 0)
        //{
        //    cv::imshow("output", frame);
        //}
        //auto total_end = std::chrono::steady_clock::now();

        //float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
        //float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
        //std::wstringstream stats_ss;
        //stats_ss << std::fixed << std::setprecision(2);
        //stats_ss << "Inference FPS: " << inference_fps << ", Total FPS: " << total_fps;
        //auto stats = stats_ss.str();
        annoFile.close();
    }

    //cv::destroyAllWindows();
    std::cout << "Finished!";
    return 0;
}