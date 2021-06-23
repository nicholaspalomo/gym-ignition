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
// ]
// TODO: Down sample the camera observation

#include <chrono>
#include <string>
#include <thread>

#include "GymIgnitionEnv.hpp"

namespace gym_ignition{

class ENVIRONMENT : public GymIgnitionEnv {

    public:

        explicit ENVIRONMENT(const std::string& resourceDir, const YAML::Node& cfg, visualizable)

};

}