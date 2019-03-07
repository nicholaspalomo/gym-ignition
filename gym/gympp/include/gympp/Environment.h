#ifndef GYMPP_ENVIRONMENT
#define GYMPP_ENVIRONMENT

#include <memory>
#include <optional>
#include <string>
#include <vector>

#include "gympp/Space.h"
#include "gympp/common.h"

namespace gympp {
    class Environment;
    using EnvironmentName = std::string;
    using EnvironmentPtr = std::shared_ptr<gympp::Environment>;

    struct State;
    using Action = data::Sample;
    using Reward = DataSupport;
    using Observation = data::Sample;
} // namespace gympp

struct gympp::State
{
    bool done;
    std::string info;
    gympp::Reward reward;
    gympp::Observation observation;
};

// TODO: https://hub.packtpub.com/openai-gym-environments-wrappers-and-monitors-tutorial/
//       These C++ and their mapping to python / julia should allow using the Wrapper method

// From https://github.com/openai/gym/blob/master/gym/core.py
class gympp::Environment
{
public:
    using Action = gympp::Action;
    using Reward = gympp::Reward;
    using Observation = gympp::Observation;
    using State = gympp::State;

    enum class RenderMode
    {
        HUMAN,
        RGB_ARRAY,
        ANSI,
    };

    using ActionSpace = gympp::spaces::Space;
    using ObservationSpace = gympp::spaces::Space;
    using ActionSpacePtr = std::shared_ptr<ActionSpace>;
    using ObservationSpacePtr = std::shared_ptr<ObservationSpace>;
    using RewardRange = gympp::Range<DataSupport>;

public:
    ActionSpacePtr action_space;
    ObservationSpacePtr observation_space;
    RewardRange reward_range;

public:
    Environment() = delete;
    virtual ~Environment() = default;

    Environment(const ActionSpacePtr aSpace,
                const ObservationSpacePtr oSpace,
                const RewardRange& rRange = {})
        : action_space(aSpace)
        , observation_space(oSpace)
        , reward_range(rRange)
    {}

    virtual std::optional<State> step(const Action& action) = 0;
    virtual std::optional<Observation> reset() = 0;
    virtual bool render(RenderMode mode) = 0;
    virtual std::vector<unsigned> seed(unsigned seed = 0) = 0;
    // TODO: close()
};

#endif // GYMPP_ENVIRONMENT