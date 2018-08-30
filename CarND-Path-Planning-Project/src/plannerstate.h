#ifndef PS_H
#define PS_H

using namespace std;

enum State {
    Follow,
    PrepareLaneChange,
    LaneChange
};

class PlannerState {
    public:
        double speed;
        State state;
        double lane;
        double lane_speed;
        double desired_lane;
};

#endif /* PS_H */
