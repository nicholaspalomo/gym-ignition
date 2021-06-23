// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

// Define the MDP for the Panda RL example

// observation space = [
//  rgb image           640 x 480 x 3   n = 921600  si = 0
//  depth image         640 x 480 x 1   n = 307200  si = 921600
//  x (end effector)    1               n = 1       si = 1228800
//  y (end effector)    1               n = 1       si = 1228801
//  z (end effector)    1               n = 1       si = 1228802
//  R (end effector)    1               n = 1       si = 1228803
//  P (end effector)    1               n = 1       si = 1228804
//  Y (end effector)    1               n = 1       si = 1228805
//  lin vel             3               n = 3       si = 1228806
//  ang vel             3               n = 3       si = 1228809
//  gripper state       1               n = 1       si = 1228810
// ]
//
// action space = [
//  x (end effector target)             n = 1       si = 0
//  y (end effector target)             n = 1       si = 1
//  z (end effector target)             n = 1       si = 2
//  R (end effector target)             n = 1       si = 3
//  P (end effector target)             n = 1       si = 4
//  Y (end effector target)             n = 1       si = 5
//  change gripper state                n = 1       si = 6
// ]
// TODO: Downsample the camera observation!

#include <stdlib.h>
#include <cstdint>
#include <set>
#include <random> // for random number generator
#include <chrono>
#include <string>
#include <thread>

#include "GymIgnitionEnv.hpp"

namespace gym_ignition{

class ENVIRONMENT : public GymIgnitionEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, visualizable) :
            GymIgnitionEnv(resourceDir, cfg),
            distribution_(0.0, 0.2),
            visualizable_(visualizable),
            uniform_dist_(0.0, 1.0),
            rand_num_gen_(std::chrono::system_clock::now().time_since_epoch().count()) {
            
            // Insert the ground plane
            const std::string ground_plane_sdf = "ground_plane/ground_plane.sdf";
            world_->insertModel(ground_plane_sdf);

            world_->setPhysicsEngine(scenario::gazebo::PhysicsEngine::Dart);

            // Open the GUI
            gazebo.gui();
            std::this_thread::sleep_for(std::chrono::seconds(3));
            gazebo.run(/*paused=*/true);

            // Insert the panda model
            const std::string panda_urdf = "panda/panda.urdf";
            world_->insertModel(/*modelFile=*/panda_urdf);
            gazebo.run(/*paused=*/true);

            panda_ = world_->getModel(/*modelName=*/"panda");

            // Set the control modes for the joints
            panda_->setJointControlMode(
                scenario::core::JointControlMode::Position,
                panda_->jointNames()
            );

            obs_dim_ = 10; // dummy value
            action_dim_ = 7; // dummy value
            extra_info_dim_ = 1; // dummy value

            obScaled_.setZero(obs_dim_);
            obDouble_.setZero(obs_dim_);
            obMean_.setZero(obs_dim_);
            obStd_.setZero(obs_dim_);
            actionMean_.setZero(action_dim_);
            actionStd_.setZero(action_dim_);
        }

        void init() final { }

        void reset() final { 

        }

        void step(const Eigen::Ref<EigenVec>& action) final {

            return 1; // dummy value
        }

        bool isTerminalState(float& terminalReward) final {
            terminalReward = 0.0; // dummy value

            return false; // dummy value
        }

        void updateExtraInfo(Eigen::Ref<EigenVec> extraInfo) {
            
        }

        void setSeed(int seed) final {
            std::srand(seed);
        }

        void observe(Eigen::Ref<EigenVec> ob) final {
            ob = obScaled_.cast<float>();
        }

    private:
        std::normal_distribution<double> distribution_;
        bool visualizable_ = false;
        std::uniform_real_distribution<double> uniform_dist_;
        std::mt19937 rand_num_gen_;

        scenario::core::ModelPtr panda_;

        Eigen::VectorXd obDouble_, obScaled_, actionMean_, actionStd_, obMean_, obStd_;
};

}