#pragma once


#include <opencv2/core/types.hpp>
#include <yaml-cpp/yaml.h>
#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>

#include <vector>

#include <Eigen/Dense>
#include <eigen3/Eigen/Core>


namespace calib_twocam {
    namespace utils {
        // load file names
        std::vector<std::string> load_file_names(const std::string &inputDir, const std::string &fileExtension);

        //
        bool parse_K(const YAML::Node& config, std::string name, cv::Mat& K);
        void parse_D(const YAML::Node& config, std::string name, cv::Mat& D_);
        cv::Size parse_image_size(const YAML::Node& config);
        void log_cvmat(const cv::Mat& mat, const std::string& name = "Mat");


        // detect
        cv::Mat genObjs(const cv::Size& boardSize, float squareSize);
        bool detectCharucoCornersAndPose(
            const cv::Size& boardSize,
            float squareSize, float markerSize,
            const cv::Mat& image, const cv::Mat& K, const cv::Mat& D, const cv::Mat& objs,
            cv::Mat& imageCopy, cv::Mat& charucoCorners, cv::Mat& charucoIds, cv::Mat& rvec, cv::Mat& tvec, double& rerror
        );
        bool detectChessCornersAndPose(
            const cv::Size& boardSize,
            const cv::Mat& image, const cv::Mat& K, const cv::Mat& D, const cv::Mat& objs,
            cv::Mat& imageCopy, cv::Mat& corners,cv::Mat& rvec, cv::Mat& tvec, double& rerror
        );

        //
        cv::Mat cvTransformPoints(const cv::Mat& src, const cv::Mat& rvec, const cv::Mat& tvec);

        // fast icp
        double fast_icp(const Eigen::MatrixXd& src, const Eigen::MatrixXd& tgt, Eigen::Matrix3d& R, Eigen::Vector3d& t);
    };
};