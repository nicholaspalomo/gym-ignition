// Copyright (C) 2021 Bosch LLC CR, North America. All rights reserved.
// This software may be modified and distributed under the terms of the
// GNU Lesser General Public License v2.1 or any later version.

#ifndef SRC_GYMIGNITION_VECENV_HPP
#define SRC_GYMIGNITION_VECENV_HPP

#include "GymIgnitionEnv.hpp"
#include "omp.h"
#include "yaml-cpp/yaml.h"

namespace gym_ignition{

template<class ChildEnvironment>
class VectorizedEnvironment {

    public:

        explicit VectorizedEnvironment(std::string resourceDir, std::string cfg) :
            resource_dir_(resourceDir) {
            cfg_ = YAML::Load(cfg);

            if(cfg_["render"].template as<bool>())
                render_ = cfg_["render"].template as<bool>();

        }

        ~VectorizedEnvironment() {
            for(auto *ptr : environments_)
                delete *ptr;
        }

        void init() {
            omp_set_num_threads(cfg_["num_threads"].template as<int>());
            num_envs_ = cfg_["num_envs"].template as<int>();

            // TODO: Set the random seed -> see GazeboSimulator.cpp. At the moment, it looks like the random seed is hardcoded in the API
            // config_->SetSeed(cfg_["seed"].template as<int>());

            for(int i = 0; i < num_envs_; i++){
                environments_.push_back(new ChildEnvironment(resource_dir_, cfg_, render_ && i == 0));
                environments_.back()->setSimulationTimeStep(cfg_["simulation_dt"].template as<double>()); // See also line 618 of GazeboSimulator.cpp
                environments_.back()->setControlTimeStep(cfg_["control_dt"].template as <double>());
            }

            // TODO: if render_ is false, hide the Gazebo gui window. Figure out how to do this

            for(int i = 0; i < num_envs_; i++){
                environment_[i]->init();
                environments_[i]->reset();
            }

            obs_dim_ = environments_[0]->getObsDim();
            action_dim_ = environments_[0]->getActionDim();
            extra_info_dim_ = environments_[0]->getExtraInfoDim();
            if(obs_dim_ == 0 || action_dim_ == 0)
                throw "[VectorizedEnvironment.hpp] Observation/action dimension must be defined in the constructor of each environment!";
        }

        // resets all environments and returns the observation
        void reset(Eigen::Ref<EigenRowMajorMat>& ob){
    #pragma omp parallel for schedule(dynamic)
            for(auto env : environments_)
                env->reset();

            observe(ob);
        }

        // retrieve the observation from the environment
        void observe(Eigen::Ref<EigenRowMajorMat>& ob){
    #pragma omp parallel for schedule(dynamic)
            for(auto env : environments_)
                env->observe(ob.row(i));
        }

        // retrieve the extra signals from the environment
        void getExtraInfo(Eigen::Ref<EigenRowMajorMat>& extraInfo){
    #pragma omp parallel for schedule(dynamic)
            for(auto env : environments_)
                env->getExtraInfo(extraInfo.row(i));
        }

        // advance the environment one step (i.e. one control_dt)
        void step(Eigen::Ref<EigenRowMajorMat>& action,
                  Eigen::Ref<EigenRowMajorMat>& ob,
                  Eigen::Ref<EigenRowMajorMat>& reward,
                  Eigen::Ref<EigenRowMajorMat>& done) {
    #pragma omp parallel for schedule(dynamic)
            for(int i = 0; i < num_envs_; i++){
                perAgentStep(i, action, ob, reward, done);
                environments_[i]->observe(ob.row(i));
            }
        }

        // TODO: Methods to start/stop recording video; show/hide the Gazebo GUI

        void close(){
            for(auto *env : environments_)
                env->close();
        }

        void isTerminalState(Eigen::Ref<EigenBoolVec>& terminalState) {
            for(int i = 0; i < num_envs_; i++){
                float terminalReward;
                terminalState[i] = environments_[i]->isTerminalState(terminalReward);
            }
        }

        void setSimulationTimeStep(double dt) {
            for (auto *env: environments_)
            env->setSimulationTimeStep(dt);
        }

        void setControlTimeStep(double dt) {
            for (auto *env: environments_)
            env->setControlTimeStep(dt);
        }

        int getObsDim() { return obs_dim_; }
        int getExtraInfoDim() { return extra_info_dim_; }
        int getActionDim() { return action_dim_; }
        int getNumOfEnvs() { return num_envs_; }

    private:
        std::string resource_dir_;
        bool render_ = false;
        YAML::Node cfg_;

        std::vector<ChildEnvironment*> environments_;
        int obs_dim_ = 0, action_dim_ = 0, extra_info_dim_ = 0;
        int num_envs_ = 1;

        std::shared_ptr<ignition::gazebo::ServerConfig> config_;

        // step() for a single environment
        inline void perAgentStep(int agentId,
                                 Eigen::Ref<EigenRowMajorMat>& action,
                                 Eigen::Ref<EigenRowMajorMat>& ob,
                                 Eigen::Ref<EigenRowMajorMat>& reward,
                                 Eigen::Ref<EigenRowMajorMat>& done) {
            reward[agentId] = environments_[agentId]->step(action.row(agentId));

            float terminal_reward = 0;
            done[agentId] = environments_[agentId]->isTerminalState(terminalReward);

            if(done[agentId]) {
                environments_[agentId]->reset();
                reward[agentId] += terminal_reward;
            }
        }
};

}

#endif //SRC_GYMIGNITION_VECENV_HPP