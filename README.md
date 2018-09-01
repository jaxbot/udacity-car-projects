# Udacity Car Nanodegree Projects

Here are all the projects I completed as part of Udacity's Self-Driving Car Nanodegree. Each project has a README that describes the project, usually with GIFs or links to videos of the end result.

## Highlights

![An animated GIF of Carla driving itself around Udacity's test track.](./Integration-Capstone-Project/capstone.gif)

[Capstone Project](Integration-Capstone-Project): Use ROS and write code that will run on Udacity's physical Carla test car. This was a group project and I owned the red light detection end-to-end, building a TF classifier using the TF Object Detection API and training a ResNet on sample traffic light datasets.

![An animation of vehicle tracking on a California highway](./CarND-Vehicle-Detection/output_images/project-output-short.gif)

[Vehicle Detection](CarND-Vehicle-Detection): Given an input video, annotate and track car locations in each frame. I wanted a smooth solution, so I cached locations between keyframes and used optical-flow to move bounding boxes so the entire frame search only had to be explored every 15 frames or so. This allows the algorithm to run in realtime.

![Path planning project, an animation of a vehicle passing slower vehicles on the highway.](./CarND-Path-Planning-Project/lane-change.gif)

[Path Planning](CarND-Path-Planning-Project): Plan paths for a vehicle on the highway to pass other vehicles, optimizing for speed while requiring jerk/braking/acceleration limitations.

![Animation of lane area being annotated on a highway](./CarND-Advanced-Lane-Lines/output_images/project-output-short.gif)

[Advanced Lane Lines](CarND-Advanced-Lane-Lines): Annotate lane driving area on a highway in a video feed. This required calibrating the camera to remove lens distortion, transforming the perspective to a top-down view, and creating a polynomial fit across lane lines.

![Animation of a car on a rough track trying to navigate](./CarND-Behavioral-Cloning-P3/output/track2.gif)

[Behavioral Cloning](./CarND-Behavioral-Cloning-P3): Record footage of driving on a video game track with steering angles annotated, then use that to train car to drive itself on that track. Like the GTA V stuff on Twitch.

![Animation of a car on a video game track following planned path points](CarND-MPC-Project/output/mpc.gif)

[MPC Project](CarND-MPC-Project): Drive a car using simulated Drive-By-Wire with latency, using cost functions to prevent overcorrecting at high speeds.

![Animation of Kalman Filter being used to localize a car on a path](./CarND-Extended-Kalman-Filter-Project/udacity.gif)

[Extended Kalman Filter Project](CarND-Extended-Kalman-Filter-Project)
