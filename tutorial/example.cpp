#include <scenario/gazebo/GazeboSimulator.h>
#include <scenario/gazebo/Joint.h>
#include <scenario/gazebo/Model.h>
#include <scenario/gazebo/World.h>

#include <chrono>
#include <string>
#include <thread>

// Some helpful tips:
// In the bashrc, put the following paths so that the simulator can find the libraries and meshes:
// export IGN_GAZEBO_RESOURCE_PATH=$IGN_GAZEBO_RESOURCE_PATH:/home/nicholas/workspace/gym-ignition/gym-ignition-models/gym_ignition_models
// export IGN_GAZEBO_SYSTEM_PLUGIN_PATH=$IGN_GAZEBO_SYSTEM_PLUGIN_PATH:/home/nicholas/workspace/gym-ignition/build/lib

int main(int argc, char* argv[])
{
    // Create the simulator
    auto gazebo = scenario::gazebo::GazeboSimulator(
        /*stepSize=*/0.001, /*rtf=*/1.0, /*stepsPerRun=*/1);

    // Initialize the simulator
    gazebo.initialize();

    // Get the default world
    auto world = gazebo.getWorld();

    // Insert the ground plane
    const std::string groundPlaneSDF = "ground_plane/ground_plane.sdf";
    world->insertModel(groundPlaneSDF);

    // Select the physics engine
    world->setPhysicsEngine(scenario::gazebo::PhysicsEngine::Dart);

    // Open the GUI
    gazebo.gui();
    std::this_thread::sleep_for(std::chrono::seconds(3));
    gazebo.run(/*paused=*/true);

    // Insert a pendulum
    const std::string pendulumURDF = "pendulum/pendulum.urdf";
    world->insertModel(/*modelFile=*/pendulumURDF);
    gazebo.run(/*paused=*/true);

    // Get the pendulum
    auto pendulum = world->getModel(/*modelName=*/"pendulum");

    // Reset the pole position
    auto pivot = pendulum->getJoint("pivot");
    auto pivotGazebo = std::static_pointer_cast<scenario::gazebo::Joint>(pivot);
    pivotGazebo->resetPosition(0.001);

    // Simulate 30 seconds
    for (size_t i = 0; i < 30.0 / gazebo.stepSize(); ++i) {
        gazebo.run();
    }

    // Close the simulator
    std::this_thread::sleep_for(std::chrono::seconds(3));
    gazebo.close();

    return 0;
}