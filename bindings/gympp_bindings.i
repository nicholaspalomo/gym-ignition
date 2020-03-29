%module gympp_bindings

%{
#define SWIG_FILE_WITH_INIT
#include "gympp/base/Common.h"
#include "gympp/base/Environment.h"
#include "gympp/base/Robot.h"
#include "gympp/base/Space.h"
#include "gympp/gazebo/GazeboEnvironment.h"
#include "gympp/gazebo/GymFactory.h"
#include "gympp/gazebo/Metadata.h"
#include "gympp/gazebo/RobotSingleton.h"
#include "scenario/gazebo/GazeboSimulator.h"
#include "scenario/gazebo/Joint.h"
#include "scenario/gazebo/Link.h"
#include "scenario/gazebo/Model.h"
#include "scenario/gazebo/World.h"
#include <cstdint>
%}

%naturalvar;

%include <stdint.i>

%include <std_array.i>
%include <std_string.i>
%include <std_vector.i>

// Convert python list to std::vector
%template(Vector_i) std::vector<int>;
%template(Vector_u) std::vector<size_t>;
%template(Vector_f) std::vector<float>;
%template(Vector_d) std::vector<double>;
%template(Vector_s) std::vector<std::string>;

// Convert python list to std::array
%template(Array3d) std::array<double, 3>;
%template(Array4d) std::array<double, 4>;
%template(Array6d) std::array<double, 6>;

%include "gympp/base/Common.h"
%template(BufferContainer_i) gympp::base::BufferContainer<int>;
%template(BufferContainer_u) gympp::base::BufferContainer<size_t>;
%template(BufferContainer_f) gympp::base::BufferContainer<float>;
%template(BufferContainer_d) gympp::base::BufferContainer<double>;
%template(get_i) gympp::base::data::Sample::get<int>;
%template(get_u) gympp::base::data::Sample::get<size_t>;
%template(get_f) gympp::base::data::Sample::get<float>;
%template(get_d) gympp::base::data::Sample::get<double>;
%template(getBuffer_i) gympp::base::data::Sample::getBuffer<int>;
%template(getBuffer_u) gympp::base::data::Sample::getBuffer<size_t>;
%template(getBuffer_f) gympp::base::data::Sample::getBuffer<float>;
%template(getBuffer_d) gympp::base::data::Sample::getBuffer<double>;

%include "optional.i"
%template(Optional_i) std::optional<int>;
%template(Optional_u) std::optional<size_t>;
%template(Optional_f) std::optional<float>;
%template(Optional_d) std::optional<double>;
%template(Optional_state) std::optional<gympp::base::State>;
%template(Optional_sample) std::optional<gympp::base::data::Sample>;

%include <std_shared_ptr.i>
%shared_ptr(gympp::base::spaces::Space)
%shared_ptr(gympp::base::spaces::Box)
%shared_ptr(gympp::base::spaces::Discrete)
%include "gympp/base/Space.h"

%shared_ptr(gympp::base::Environment)
%shared_ptr(scenario::gazebo::GazeboSimulator)
%shared_ptr(gympp::gazebo::GazeboEnvironment)
%include "ignition/common/SingletonT.hh"
%ignore ignition::common::SingletonT<gympp::gazebo::GymFactory>::myself;
%template(GymFactorySingleton) ignition::common::SingletonT<gympp::gazebo::GymFactory>;

%include "gympp/base/Environment.h"
%include "scenario/gazebo/GazeboSimulator.h"
%include "gympp/gazebo/GazeboEnvironment.h"

%extend gympp::base::Robot {
    bool setdt(const double dt) {
        return $self->setdt(std::chrono::duration<double>(dt));
    }

    double dt() const {
        return $self->dt().count();
    }
}

%include "weak_ptr.i"
%shared_ptr(gympp::base::Robot)
%template(RobotWeakPtr) std::weak_ptr<gympp::base::Robot>;

%ignore gympp::base::Robot::dt;
%ignore gympp::base::Robot::setdt(const StepSize&);
%include "gympp/base/Robot.h"
%template(Vector_contact) std::vector<gympp::base::ContactData>;

%inline %{
    std::shared_ptr<gympp::gazebo::GazeboEnvironment> envToGazeboEnvironment(gympp::base::EnvironmentPtr env) {
        return std::dynamic_pointer_cast<gympp::gazebo::GazeboEnvironment>(env);
    }

    std::shared_ptr<scenario::gazebo::GazeboSimulator> envToGazeboWrapper(gympp::base::EnvironmentPtr env) {
        return std::dynamic_pointer_cast<scenario::gazebo::GazeboSimulator>(env);
    }
%}

%include "gympp/gazebo/Metadata.h"
%include "gympp/gazebo/GymFactory.h"
%include "gympp/gazebo/RobotSingleton.h"

%shared_ptr(scenario::gazebo::Joint)
%shared_ptr(scenario::gazebo::Link)
%shared_ptr(scenario::gazebo::Model)
%shared_ptr(scenario::gazebo::World)

%include "scenario/gazebo/Joint.h"
%include "scenario/gazebo/Link.h"
%include "scenario/gazebo/Model.h"
%include "scenario/gazebo/World.h"
