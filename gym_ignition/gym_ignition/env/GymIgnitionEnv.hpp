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



}
