// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#ifndef SRC_GYMIGNITIONENV_HPP
#define SRC_GYMIGNITIONENV_HPP

#include <vector>
#include <memory>
#include <unordered_map>
#include <Eigen/Core>

#include "yaml-cpp/yaml.h"

#include <scenario/gazebo/GazeboSimulator.h>
#include <scenario/gazebo/Joint.h>
#include <scenario/gazebo/Model.h>
#include <scenario/gazebo/World.h>

#define __RSG_MAKE_STR(x) #x
#define _RSG_MAKE_STR(x) __RSG_MAKE_STR(x)
#define RSG_MAKE_STR(x) _RSG_MAKE_STR(x)

#define READ_YAML(a, b, c) RSFATAL_IF(!c, "Node "<<RSG_MAKE_STR(c)<<" doesn't exist") \
                           b = c.as<a>();

namespace gym_ignition {

using EigenRowMajorMat=Eigen::Matrix<float, -1, -1, Eigen::RowMajor>;
using EigenVec=Eigen::Matrix<float, -1, 1>;
using EigenBoolVec=Eigen::Matrix<bool, -1, 1>;

class GymIgnitionEnv(){

    public:

        explicit GymIgnitionEnv(std::string resourceDir, const YAML::Node& cfg) :
            resource_dir_(std::move(resourceDir)), cfg_(cfg) {

                // Create the simulator
                gazebo_ = std::make_unique(scenario::gazebo::Simulator(cfg_["step_size"], cfg_["rtf"], cfg_["steps_per_run"]))
                
                // Initialize the simulator
                gazebo_->initialize();

                // Get a pointer to the world
                world_ = gazebo_->getWorld();
            }

        virtual ~GymIgnitionEnv() = default;

        //// Implement the following methods as part of the MDP ////
        virtual void init() = 0;
        virtual void reset() = 0;
        virtual void setSeed(int seed) = 0;
        virtual void observe(Eigen::Ref<EigenVec> ob) = 0;
        virtual float step(const Eigen::Ref<EigenVec>& action) = 0;
        virtual bool isTerminalState(float& terminalReward) = 0;
        virtual void updateExtraInfo(Eigen::Ref<EigenVec> extraInfo) {};
        ////////////////////////////////////////////////////////////

        //// Option methods to implement ////
        virtual void close() {};
        /////////////////////////////////////

        int getObsDim() { return obs_dim_; }
        int getActionDim() { return action_dim_; }
        int getExtraInfoDim() { return extra_info_dim_; }

        void turnOnVisualization() { visualize_this_step_ = true; }
        void turnOffVisualization() { visualize_this_step_ = false; }
        void setControlTimeStep(double dt) { control_dt_ = dt; }
        double getControlTimeStep() { return control_dt_; }

        void setSimulationTimeStep(double dt){
            sim_dt_ = dt;

            sdf::Root root;
            sdf::ElementPtr sdfElement = gazebo_->sdfElement;
            auto errors = root.LoadSdfString(sdfElement->ToString(""));
            assert(errors.empty()); // TODO
            for (size_t worldIdx = 0; worldIdx < root.WorldCount(); ++worldIdx) {
                if (!utils::updateSDFPhysics(root,
                                            gazebo_->physics.dt,
                                            gazebo_->physics.rtf,
                                            /*realTimeUpdateRate=*/-1,
                                            worldIdx)) {
                    sError << "Failed to set physics profile" << std::endl;
                    throw "[GymIgnitionEnv.hpp] Fatal error.";
                }
            }

        }
        double getSimulationTimeStep() { return sim_dt_; }

        std::unordered_map<std::string, float> extra_info_;

    protected:

        std::string resource_dir_;
        YAML::Node cfg_;

        std::unique_ptr<scenario::gazebo::GazeboSimulator> gazebo_;
        std::shared_ptr<scenario::gazebo::World> world_;

        int obs_dim_ = 0, action_dim_ = 0, extra_info_dim_ = 0;

        double sim_dt_ = 0.001, control_dt_ = 0.001;
        bool visualize_this_step_ = false;
};

}

#endif // SRC_GYMIGNITIONENV_HPP
