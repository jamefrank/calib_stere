#pragma once

#include <ceres/ceres.h>
#include <ceres/rotation.h>
#include <eigen3/Eigen/Core>


struct TwoCamReprojectionError
{
    TwoCamReprojectionError(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        const Eigen::Matrix3d& K,
        bool bcam1_to_cam2
        )
        : point_target_(obj), corner_cam2_(corner), K_(K), bcam1_to_cam2_(bcam1_to_cam2) {}

    // target2cam1,cam2gripper: rvec+tvec
    template <typename T>
    bool operator()(const T* const target2cam1, const T* const cam2gripper, const T* const KFxy, T* residuals) const{
        //
        T p_target[3];
        p_target[0] = T(point_target_(0));
        p_target[1] = T(point_target_(1));
        p_target[2] = T(point_target_(2));
        // 
        T p_cam1[3];
        ceres::AngleAxisRotatePoint(target2cam1, p_target, p_cam1);
        p_cam1[0] += target2cam1[3];
        p_cam1[1] += target2cam1[4];
        p_cam1[2] += target2cam1[5];
        //
        T p_cam2[3];
        if(bcam1_to_cam2_) {
            ceres::AngleAxisRotatePoint(cam2gripper, p_cam1, p_cam2);
            p_cam2[0] += cam2gripper[3];
            p_cam2[1] += cam2gripper[4];
            p_cam2[2] += cam2gripper[5];
        }
        else {
            p_cam2[0] = p_cam1[0] - cam2gripper[3];
            p_cam2[1] = p_cam1[1] - cam2gripper[4];
            p_cam2[2] = p_cam1[2] - cam2gripper[5];
            T rotation[3] = {T(-1) * cam2gripper[0], T(-1) * cam2gripper[1], T(-1) * cam2gripper[2]};
            ceres::AngleAxisRotatePoint(rotation, p_cam2, p_cam2);
        }

         //
        T u = p_cam2[0] / p_cam2[2];
        T v = p_cam2[1] / p_cam2[2];

        T fx = KFxy[0];
        T cx = T(K_(0,2));
        T p_x = fx*u + cx;

        T fy = KFxy[1];
        T cy = T(K_(1,2));
        T p_y = fy*v + cy;
        
        residuals[0] = p_x - T(corner_cam2_(0));
        residuals[1] = p_y - T(corner_cam2_(1));


        return true;
    }

    static ceres::CostFunction *Create(
        const Eigen::Vector3d& obj, 
        const Eigen::Vector2d& corner, 
        const Eigen::Matrix3d& K,
        bool bcam1_to_cam2)
    {
        return (new ceres::AutoDiffCostFunction<TwoCamReprojectionError, 2, 6, 6, 2>(new TwoCamReprojectionError(obj, corner, K, bcam1_to_cam2)));
    }

    // members
    Eigen::Vector3d point_target_;
    Eigen::Vector2d corner_cam2_;
    Eigen::Matrix3d K_;
    bool bcam1_to_cam2_;
};