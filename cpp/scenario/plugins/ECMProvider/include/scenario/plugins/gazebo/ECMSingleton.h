/*
 * Copyright (C) 2020 Istituto Italiano di Tecnologia (IIT)
 * All rights reserved.
 *
 * This project is dual licensed under LGPL v2.1+ or Apache License.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * This software may be modified and distributed under the terms of the
 * GNU Lesser General Public License v2.1 or any later version.
 *
 * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * *
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef SCENARIO_PLUGINS_GAZEBO_ECMSINGLETON_H
#define SCENARIO_PLUGINS_GAZEBO_ECMSINGLETON_H

#include "ignition/gazebo/EntityComponentManager.hh"
#include "ignition/gazebo/EventManager.hh"

#include <memory>

namespace scenario {
    namespace plugins {
        namespace gazebo {
            class ECMSingleton;
        } // namespace gazebo
    } // namespace plugins
} // namespace scenario

class scenario::plugins::gazebo::ECMSingleton
{
public:
    ECMSingleton();
    ~ECMSingleton() = default;

    ECMSingleton(ECMSingleton&) = delete;
    void operator=(const ECMSingleton&) = delete;

    static ECMSingleton& get();

    void clean();
    bool valid() const;

    ignition::gazebo::EventManager* getEventManager() const;
    ignition::gazebo::EntityComponentManager* getECM() const;

    bool storePtrs(ignition::gazebo::EntityComponentManager* ecm,
                   ignition::gazebo::EventManager* eventMgr);

private:
    class Impl;
    std::unique_ptr<Impl> pImpl;
};

#endif // SCENARIO_PLUGINS_GAZEBO_ECMSINGLETON_H
