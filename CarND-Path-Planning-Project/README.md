# CarND-Path-Planning-Project
Self-Driving Car Engineer Nanodegree Program

[state-machine]: ./state-machine.png "State machine"
[lane-change]: ./lane-change.gif "GIF of a lane change in the simulator"

![GIF of a lane change in the simulator][lane-change]

## Reflection: Model documentation

The method for generating points in this model is based on the advice given in the project walkthrough video. However, many changes were made to fine-tune performance and answer acceleration and jerk thresholds were maintained while traveling for miles without incident. In testing, the car has been able to travel more than 25 miles without incident; the only caveat comes from when vehicles in the simulator cut off our car, in which case there exists no choice but to slam on the brakes and incur a jerk violation.

The anchor points are spaced 40m apart, 10m further than the classroom example, which gives the car smoother (less jerky) movements with the tradeoff that the vehicle may briefly kiss lane lines during sharper curves. This value is increased to 50m during lane changes to ensure smoother curves.

The velocity of the vehicle in the planner is calculated based on the expected velocity at the simulator's previous points, not necessarily the currently given car speed. This allows us to space path points apart with full control over the acceleration curve. In normal following, velocity shifts are capped at 0.25mph +/- between points. During lane changes, this is reduced to 0.15mph to limit jerkiness while moving laterally.

The path generation is limited to generating points for a given speed and lane position. The determination of safe speed and lane position is predetermined by the planner using a state machine.

### State machine

The planner has three defined states: Follow, PrepareLaneChange, and LaneChange, which transition in the following way:

![State machine][state-machine]

The default state is Follow, in which the car will follow the speed of the lead car, or otherwise proceed at the speed limit. If the car becomes too close (less than 30m away), the car will reduce speed slightly to grow a gap. This gives us ample stopping distance without having to slam on the brakes uncomfortably.

If the lead car is traveling below the speed limit, the planner will transition from the Follow state to the PrepareLaneChange state, and keep track of the current fastest lane adjacent to the car's path. In this state, the car will immediately transition to making a lane change if the following conditions are met, indicating a safe lane change is possible:

* The desired lane's lead car is at least 15m in front
* The desired lane's tailing car is at least 15m behind
* The current lane's lead car is at least 15m in front, such that we will not risk hitting it while
* There is less than a 15mph difference in lane speeds, or the vehicle is traveling at a low enough speed that jerk will not be a concern.

While these conditions are not met, the car in PrepareLaneChange state will drive similarly to the Follow state, except that it will slow down to the desired lane's speed if necessary to initiate the lane change.

Once transitioning to the LaneChange state, the anchor generation starts plotting points in the desired lane instead of the correct lane, and waypoints are spaced slightly further apart for a smoother curve; in addition, the velocity change rate is reduced to prevent heavy acceleration or braking during the lane change. Most importantly, the desired lane is now static, and the planner will no longer check for faster lanes during the maneuvar. This prevents the car from starting a lane change and swinging back out aggressively when other vehicles speed up / slow down and make the other lane more desirable, but it does come with the drawback that the vehicle may complete a full lane change, then immediately lane change back to the previous lane, depending on the behavior of the other vehicles on the road.

The planner automatically transitions back to Follow any time the current lane is the most desirable lane (i.e. is the fastest), and this happens both when a lane change is completed and when a lane change is no longer necessary.
