#include "utils.h"

#include <boost/filesystem.hpp>

#include "spdlog/spdlog.h"
#include <cassert>
#include <opencv2/aruco/charuco.hpp>
#include <Eigen/SVD>
std::vector<std::string> calib_twocam::utils::load_file_names(const std::string &inputDir, const std::string &fileExtension) {
    std::vector<std::string> imageFilenames;

    boost::filesystem::directory_iterator itr;
    for (boost::filesystem::directory_iterator itr(inputDir); itr != boost::filesystem::directory_iterator(); ++itr) {
        if (!boost::filesystem::is_regular_file(itr->status())) {
            continue;
        }
        std::string filename = itr->path().filename().string();
        // check if file extension matches
        if (filename.compare(filename.length() - fileExtension.length(), fileExtension.length(), fileExtension) != 0) {
            continue;
        }

        imageFilenames.push_back(itr->path().string());
    }

    std::sort(imageFilenames.begin(), imageFilenames.end());

    return imageFilenames;
}

cv::Mat calib_twocam::utils::genObjs(const cv::Size& boardSize, float squareSize) {
    float squareLength = squareSize / 1000.0f;
	cv::Mat objs(boardSize.width*boardSize.height,1,CV_32FC3);
	for(int i=0;i<objs.rows;i++)
	{
		int idx = i % boardSize.width;
		int idy = i / boardSize.width;
		objs.at<cv::Vec3f>(i)[0] =(idx + 1) * squareLength;
		objs.at<cv::Vec3f>(i)[1] =(idy + 1) * squareLength;
		objs.at<cv::Vec3f>(i)[2] =0;
	}
	return objs;
}

bool calib_twocam::utils::detectCharucoCornersAndPose(
            const cv::Size& boardSize,
            float squareSize, float markerSize,
            const cv::Mat& image, const cv::Mat& K, const cv::Mat& D, const cv::Mat& objs,
            cv::Mat& imageCopy, cv::Mat& charucoCorners, cv::Mat& charucoIds, cv::Mat& rvec, cv::Mat& tvec, double& rerror
        ) {
    int squaresX = boardSize.width + 1;
	int squaresY = boardSize.height + 1;
	float squareLength = squareSize / 1000.0f;
	float markerLength = markerSize / 1000.0f;
	cv::Ptr<cv::aruco::Dictionary> dictionary_ = cv::aruco::getPredefinedDictionary(cv::aruco::DICT_4X4_250);
	cv::Ptr<cv::aruco::CharucoBoard> board_ = cv::aruco::CharucoBoard::create(squaresX, squaresY, squareLength, markerLength, dictionary_);

	cv::Mat gray, undistImg;
	cv::undistort(image, undistImg, K, D);
    undistImg.copyTo(imageCopy);
    cv::cvtColor(undistImg, gray, cv::COLOR_BGR2GRAY);
    // cv::equalizeHist(gray, gray);
	// cv::Mat blur_usm;
    // cv::GaussianBlur(gray, blur_usm, cv::Size(0, 0), 25);
    // cv::addWeighted(gray, 1.5, blur_usm, -0.5, 0, gray);

	cv::Ptr<cv::aruco::DetectorParameters> parameters = cv::aruco::DetectorParameters::create();
    std::vector<int> marker_ids;
    std::vector<std::vector<cv::Point2f>> marker_corners, marker_rejected;

    cv::aruco::detectMarkers(gray, dictionary_, marker_corners, marker_ids, parameters, marker_rejected);

    cv::aruco::interpolateCornersCharuco(marker_corners, marker_ids, gray, board_, charucoCorners, charucoIds);
    if(charucoIds.rows == boardSize.width * boardSize.height){
        
        bool valid = cv::aruco::estimatePoseCharucoBoard(charucoCorners, charucoIds, board_, K, cv::Mat(), rvec, tvec);
        if(valid){
            // calc error
            cv::Mat imgpoints;
            cv::projectPoints(objs, rvec, tvec, K, cv::Mat(), imgpoints);
            rerror = 0;
            for(int i=0;i<imgpoints.rows;i++)
            {
                cv::Vec2f e = imgpoints.at<cv::Vec2f>(i) - charucoCorners.at<cv::Vec2f>(i);
                rerror += std::sqrt(e[0]*e[0]+e[1]*e[1]);
            }
            rerror /= imgpoints.rows;
            
            //
			cv::aruco::drawDetectedMarkers(imageCopy, marker_corners, marker_ids);
			cv::aruco::drawDetectedCornersCharuco(imageCopy, charucoCorners, charucoIds, cv::Scalar(255, 0, 0));
			cv::drawFrameAxes(imageCopy, K, cv::Mat(), rvec, tvec, 0.1f);
			return true;
        }
    }
    else
        spdlog::error("Charuco Board detect failed (num corners): {}", charucoIds.rows);
  
	return false;
}

bool calib_twocam::utils::detectChessCornersAndPose(
            const cv::Size& boardSize,
            const cv::Mat& image, const cv::Mat& K, const cv::Mat& D, const cv::Mat& objs,
            cv::Mat& imageCopy, cv::Mat& corners,cv::Mat& rvec, cv::Mat& tvec, double& rerror
        ) {
    cv::Mat gray, undistImg;
	cv::undistort(image, undistImg, K, D);
    undistImg.copyTo(imageCopy);
    cv::cvtColor(undistImg, gray, cv::COLOR_BGR2GRAY);

	bool success = cv::findChessboardCorners(gray, boardSize, corners, cv::CALIB_CB_ADAPTIVE_THRESH + cv::CALIB_CB_NORMALIZE_IMAGE);
	if(success && corners.rows==boardSize.width*boardSize.height){
		cv::cornerSubPix(gray, corners, cv::Size(3,3), cv::Size(-1, -1), cv::TermCriteria(cv::TermCriteria::EPS + cv::TermCriteria::COUNT, 30, 0.1));
		bool valid = cv::solvePnP(objs, corners, K, cv::Mat(), rvec, tvec);
		if(valid){
			// calc error
            cv::Mat imgpoints;
            cv::projectPoints(objs, rvec, tvec, K, cv::Mat(), imgpoints);
            rerror = 0;
            for(int i=0;i<imgpoints.rows;i++)
            {
                cv::Vec2f e = imgpoints.at<cv::Vec2f>(i) - corners.at<cv::Vec2f>(i);
                rerror += std::sqrt(e[0]*e[0]+e[1]*e[1]);
            }
            rerror /= imgpoints.rows;

			cv::drawChessboardCorners(imageCopy, boardSize, corners, success);
			cv::drawFrameAxes(imageCopy, K, cv::Mat(), rvec, tvec, 0.1f);
			return true;
		}
	}

	return false;
}

bool calib_twocam::utils::parse_K(const YAML::Node& config, std::string name, cv::Mat& K) {
    if(config[name]){
        if(config[name].IsSequence()){
            int num = config[name].size();
            if(9==num || 3==num){
                std::vector<float> tmp;   
                if(9 == num){
                    for(int i=0; i<num; i++){
                        float value = config[name][i].as<float>();
                        tmp.push_back(value);
                    }
                }
                if(3 == num){
                    for(int i=0; i<num; i++){
                        for (int j = 0; j < config[name][i].size(); ++j) {
                            float value = config[name][i][j].as<float>();
                            tmp.push_back(value);
                        }
                    }
                }
                K = cv::Mat(3, 3, CV_32F, tmp.data()).clone();
                return true;
            }
            else
                spdlog::error("3x3 or 9 supported");
        }
        else
            spdlog::error("[{0}] not list", name);
    }
    else{
        spdlog::error("{0} not exists", name);
    }
    
    return false;
}

void calib_twocam::utils::parse_D(const YAML::Node& config, std::string name, cv::Mat& D_) {
    std::vector<float> D;   //TODO
    for (std::size_t i = 0; i < config["D"].size(); ++i) {
        float value = config["D"][i].as<float>();
        D.push_back(value);
    }
    D_ = cv::Mat(1, D.size(), CV_32F, D.data()).clone();
}

cv::Size calib_twocam::utils::parse_image_size(const YAML::Node& config) {
    int width = config["width"].as<int>();
    int height = config["height"].as<int>();

    return cv::Size(width, height);
}

void calib_twocam::utils::log_cvmat(const cv::Mat& mat, const std::string& name) {
    std::ostringstream oss;
    oss << name << " = " << std::endl << mat << std::endl;
    spdlog::info("{}", oss.str());
}

cv::Mat calib_twocam::utils::cvTransformPoints(const cv::Mat& src, const cv::Mat& rvec, const cv::Mat& tvec) {
    CV_Assert(src.type() == CV_32FC3 || src.type() == CV_32FC1);

    cv::Mat R;
    cv::Rodrigues(rvec, R);

    cv::Mat pts = src.reshape(1, pts.rows);
    pts.convertTo(pts, CV_64F);
    CV_Assert(pts.cols == 3);



    cv::Mat transformed;
    transformed = pts * R.t();      // 旋转

    cv::Mat tvec_row = tvec.reshape(1, 1); // 确保是 1x3 行向量
    cv::Mat tvec_repeated;
    cv::repeat(tvec_row, transformed.rows, 1, tvec_repeated); // 复制成 Nx3
    transformed += tvec_repeated;  

    return transformed; //N*3
}

double calib_twocam::utils::fast_icp(const Eigen::MatrixXd& src, const Eigen::MatrixXd& tgt, Eigen::Matrix3d& R, Eigen::Vector3d& t) {
    //
    Eigen::MatrixXd P = src;
    Eigen::MatrixXd Q = tgt;
    
    //
    assert(3 == P.cols());
    assert(3 == Q.cols());
    assert(P.rows() == Q.rows());

    //
    int nums = P.rows();

    Eigen::Vector3d m1(0, 0, 0);
    Eigen::Vector3d m2(0, 0, 0);
    for (int i = 0; i < nums; i++) {
        m1 += P.row(i).transpose();
        m2 += Q.row(i).transpose();
    }

    m1 = m1 / (nums);
    m2 = m2 / (nums);

    for (int i = 0; i < nums; i++) {
        P.row(i) = P.row(i) - m1.transpose();
        Q.row(i) = Q.row(i) - m2.transpose();
    }

    Eigen::MatrixXd cov = P.transpose() * Q;
    Eigen::JacobiSVD<Eigen::MatrixXd> svd(cov, Eigen::ComputeFullU | Eigen::ComputeFullV);

    double det = (svd.matrixV() * svd.matrixU().transpose()).determinant();
    if (det > 0)
        det = 1.0;
    else
        det = -1.0;
    Eigen::Matrix3d I = Eigen::Matrix3d::Identity(3, 3);
    I(2, 2) = det;
    R = svd.matrixV() * I * svd.matrixU().transpose();
    t = m2 - R * m1;

    // std::cout << "[P -> Q] init R:" << std::endl;
    // std::cout << R << std::endl;
    // std::cout << "[P -> Q] init t:" << std::endl;
    // std::cout << t << std::endl;

    // calc error
    auto aligned = (src * R.transpose()).rowwise() + t.transpose();
    auto error_mat = aligned  - tgt;
    double mean_error = Eigen::VectorXd(error_mat.rowwise().norm()).mean();

    return mean_error;
}
