#include "cmdline.h"
#include "utils.h"

#include "ba_twocam.h"

#include <cassert>
#include <opencv2/calib3d.hpp>
#include <opencv2/core.hpp>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/matx.hpp>
#include <opencv2/core/types.hpp>
#include <opencv2/imgproc.hpp>
#include <spdlog/spdlog.h>
#include <boost/filesystem.hpp>
#include <opencv2/core/eigen.hpp>

#include <string>
#include <vector>
using namespace std;

int main(int argc, char *argv[])
{
    spdlog::info("Welcome to use calib_stere calib tool!");

    cmdline::parser parser;
    parser.add<string>("left", 'l', "Source Input directory containing images + config.yaml", true, "");
    parser.add<string>("right", 'r', "Target Input directory containing images + config.yaml", true, "");
    parser.add<string>("extension", 'e', "File extension of images", false, ".png");
    parser.add<string>("output", 'o', "Output directory containing calib results", true, "");

    parser.add<string>("board-type", 't', "board type", false, "charuco", cmdline::oneof<string>("charuco", "chess"));
    parser.add<int>("board-width", 'w', "Number of inner corners on the chessboard pattern in x direction", false, 3);
    parser.add<int>("board-height", 'h', "Number of inner corners on the chessboard pattern in y direction", false, 2);
    parser.add<double>("square-size", 's', "Size of one square in mm", false, 36.0);
    parser.add<double>("marker-size", 'm', "Size of one square in mm", false, 27.0);

    parser.add("verbose", '\0', "verbose when calib");

    parser.parse_check(argc, argv);
    bool verbose = parser.exist("verbose");

    std::string source_dir = parser.get<string>("left");
    std::string target_dir = parser.get<string>("right");
    std::string output_dir = parser.get<string>("output");
    std::string fileExtension = parser.get<std::string>("extension");
    std::string outputDir = parser.get<std::string>("output");


    //
    std::vector<std::string> source_file_names = calib_twocam::utils::load_file_names(source_dir, fileExtension);
    std::vector<std::string> target_file_names = calib_twocam::utils::load_file_names(target_dir, fileExtension);
    assert(source_file_names.size() == target_file_names.size());

    // parse config
    std::string source_config_file = source_dir + "/config.yaml";
    YAML::Node source_config = YAML::LoadFile(source_config_file);
    cv::Mat source_K, source_D;
    calib_twocam::utils::parse_K(source_config, "K", source_K);
    calib_twocam::utils::parse_D(source_config, "D", source_D);
    calib_twocam::utils::log_cvmat(source_K, "source_K");
    calib_twocam::utils::log_cvmat(source_D, "source_D");
    cv::Size source_img_size = calib_twocam::utils::parse_image_size(source_config);

    std::string target_config_file = target_dir + "/config.yaml";
    YAML::Node target_config = YAML::LoadFile(target_config_file);
    cv::Mat target_K, target_D;
    calib_twocam::utils::parse_K(target_config, "K", target_K);
    calib_twocam::utils::parse_D(target_config, "D", target_D);
    calib_twocam::utils::log_cvmat(target_K, "target_K");
    calib_twocam::utils::log_cvmat(target_D, "target_D");
    cv::Size target_img_size = calib_twocam::utils::parse_image_size(target_config);
    CV_Assert(source_img_size == target_img_size);

    // detect corners
    std::vector<std::vector<cv::Vec2f>> all_src_corners_2d;
    std::vector<std::vector<cv::Vec2f>> all_tgt_corners_2d;
    std::vector<std::vector<cv::Vec3f>> all_obj_points_3d;

    std::string board_type = parser.get<std::string>("board-type");
    cv::Size boardSize;
    boardSize.width = parser.get<int>("board-width");
    boardSize.height = parser.get<int>("board-height");
    float squareSize = parser.get<double>("square-size");
    float markerSize = parser.get<double>("marker-size");

    cv::Mat objs = calib_twocam::utils::genObjs(boardSize, squareSize);

    for(int i = 0; i < source_file_names.size(); i++) {
        //
        cv::Mat source_image = cv::imread(source_file_names[i], -1);
        cv::Mat target_image = cv::imread(target_file_names[i], -1);

        //
        bool source_bsuc = false;
        double source_rerror = 0;
        cv::Mat source_imageCopy, source_corners, source_charucoIds, source_rvec, source_tvec;
        if("charuco" == board_type){
            source_bsuc = calib_twocam::utils::detectCharucoCornersAndPose(
                boardSize, squareSize, markerSize, 
                source_image, source_K, source_D, objs, 
                source_imageCopy, source_corners, source_charucoIds, 
                source_rvec, source_tvec, source_rerror);
        }
        else if("chess" == board_type){
            source_bsuc = calib_twocam::utils::detectChessCornersAndPose(
                boardSize, source_image, 
                source_K, source_D, objs, 
                source_imageCopy, source_corners, source_rvec, source_tvec, source_rerror);
        }

        bool target_bsuc = false;
        double target_rerror = 0;
        cv::Mat target_imageCopy, target_corners, target_charucoIds, target_rvec, target_tvec;
        if("charuco" == board_type){
            target_bsuc = calib_twocam::utils::detectCharucoCornersAndPose(
                boardSize, squareSize, markerSize, 
                target_image, target_K, target_D, objs, 
                target_imageCopy, target_corners, target_charucoIds, 
                target_rvec, target_tvec, target_rerror);
        }
        else if("chess" == board_type){
            target_bsuc = calib_twocam::utils::detectChessCornersAndPose(
                boardSize, target_image, 
                target_K, target_D, objs, 
                target_imageCopy, target_corners, target_rvec, target_tvec, target_rerror);
        }

        if(source_bsuc && target_bsuc) { 
            boost::filesystem::path source_filepath(source_file_names[i]);
            spdlog::info("{} source charuco corners rerror: {:.2f}", source_filepath.filename().string(), source_rerror);
            boost::filesystem::path target_filepath(target_file_names[i]);
            spdlog::info("{} target charuco corners rerror: {:.2f}", target_filepath.filename().string(), target_rerror);

            // 
            std::vector<cv::Vec2f> src_corners_2d, tgt_corners_2d;
            std::vector<cv::Vec3f> obj_points_3d;

            for(int idx=0; idx<source_corners.rows;idx++){
                src_corners_2d.push_back(cv::Vec2f(source_corners.at<cv::Point2f>(idx,0).x, source_corners.at<cv::Point2f>(idx,0).y));
            }
            all_src_corners_2d.push_back(src_corners_2d);

            for(int idx=0; idx<target_corners.rows;idx++) {
                tgt_corners_2d.push_back(cv::Vec2f(target_corners.at<cv::Point2f>(idx,0).x, target_corners.at<cv::Point2f>(idx,0).y));
            }
            all_tgt_corners_2d.push_back(tgt_corners_2d);

            for(int idx=0; idx<objs.rows;idx++) {
                obj_points_3d.push_back(objs.at<cv::Vec3f>(idx));
            }
            all_obj_points_3d.push_back(obj_points_3d);
            

            //
            if (verbose){
                std::string src_out_file = outputDir + "/src_" + source_filepath.filename().string();
                cv::imwrite(src_out_file, source_imageCopy);
                std::string tgt_out_file = outputDir + "/tgt_" + target_filepath.filename().string();
                cv::imwrite(tgt_out_file, target_imageCopy);
            }
        }
    }
    spdlog::info("find pairs: {}", all_obj_points_3d.size());



    // stere
    cv::Mat stere_R, stere_T, stere_F, stere_E;
    cv::TermCriteria criteria = cv::TermCriteria(cv::TermCriteria::COUNT+cv::TermCriteria::EPS, 100, 1e-5); 
    cv::Mat tmpD = (cv::Mat_<float>(1, 4)<<0,0,0,0);
    double ret = cv::stereoCalibrate(all_obj_points_3d, all_src_corners_2d, all_tgt_corners_2d, source_K, 
        tmpD, target_K, tmpD, source_img_size, stere_R, stere_T, stere_E, stere_F,cv::CALIB_FIX_INTRINSIC, criteria);
    calib_twocam::utils::log_cvmat(stere_R, "stere_R");
    calib_twocam::utils::log_cvmat(stere_T, "stere_T");
    calib_twocam::utils::log_cvmat(stere_E, "stere_E");
    calib_twocam::utils::log_cvmat(stere_F, "stere_F");
    spdlog::info("stereoCalibrate error: {}", ret);

    //
    cv::Mat R1, R2, P1, P2, Q;
    cv::stereoRectify(source_K, cv::Mat(), target_K, cv::Mat(), source_img_size, stere_R, stere_T, R1, R2, P1, P2, Q);
    cv::Mat left_map1, left_map2;
    cv::initUndistortRectifyMap(source_K, source_D, R1, P1, source_img_size, CV_16SC2, left_map1, left_map2);
    cv::Mat right_map1, right_map2;
    cv::initUndistortRectifyMap(target_K, target_D, R2, P2, source_img_size, CV_16SC2, right_map1, right_map2);

    if(verbose) {
        for(int i=0; i<source_file_names.size(); i++){
            cv::Mat left_image = cv::imread(source_file_names[i], -1);
            cv::Mat right_image = cv::imread(target_file_names[i], -1);

            cv::Mat left_image_rectified, right_image_rectified;
            cv::remap(left_image, left_image_rectified, left_map1, left_map2, cv::INTER_CUBIC);
            cv::remap(right_image, right_image_rectified, right_map1, right_map2, cv::INTER_CUBIC);

            cv::Mat result;
            cv::hconcat(left_image_rectified, right_image_rectified, result);
            int line_y = 10;
            int line_spaceing = 50;
            while (line_y < result.rows) {
                cv::line(result, cv::Point(0, line_y), cv::Point(result.cols, line_y), cv::Scalar(0, 0, 255), 2);
                line_y += line_spaceing;
            }
            boost::filesystem::path filepath(source_file_names[i]);
            std::string out_file = outputDir + "/" + filepath.filename().string();
            cv::imwrite(out_file, result);
            spdlog::info("Writing {}", out_file);
        }
    }

    return 0;
}