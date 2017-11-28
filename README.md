# CarND-Controls-PID

Self-Driving Car Engineer Nanodegree Program

---

## Reflection

### Describe the effect each of the P, I, D components had in your implementation

As expected, the P component affected how aggressively the vehicle steered when attempting to correct for crosstrack errors. Increasing this value would increase the responsiveness around turns (especially at high velocities, i.e. >50mph) but also increase the amount of overcorrecting and oscillations that occurred. Adjusting this parameter always required tweaking the D component, which affected the countersteering as the crosstrack error approached zero. This is again as expected. The I component did not appear to affect much of anything, and I suspect little to no systematic bias exists in the simulator.

My final parameters at safe speeds can be seen in the GIF below:

<img src="examples/safe-driving.gif" alt="CarND simulator running a car that mostly stays in the middle of the lane lines.">

The affect of the P and D parameters were, as noted above, most noticeable at high speeds. Increasing P would increase the responsiveness in sharp turns (which is all of them at 90mph), but also greatly increase oscillations from overcorrecting. Increasing D would help with this, but could also end up reducing the sharpness of a correction throughout a turn. These parameters may need to be adjusted in respect to the velocity of the vehicle, or whether or not a curve is being taken.

My project in reckless driving mode (RECKLESS = true, MAX_THROTTLE = 1.0) can be seen below.

<img src="examples/reckless.gif" alt="CarND simulator running a car traveling approximately 90mph and veering within lane lines, though mostly succeeding at staying on the road.">

In both examples, the PID throttle control can be seen. In my case, I choose to subject the error from the max throttle, but scale the aggressiveness of the error by the velocity of the vehicle. This ensured that the vehicle would brake hard when needing to correct for large crosstrack errors at high speeds, but would still allow acceleration of the vehicle at low, safe speeds even if the car was off-center.

### Describe how the final hyperparameters were chosen.

I first attempted to use the twiddle method to adjust the hyperparameters P/I/D, but after implementing the twiddle algorithm, I found that tuning was difficult to do in a manner that was time effective, since errors taken over anything but the entire course was far too subject to noise.

I instead resorted to manually tuning the parameters and observing the effects at high speeds. I settled on the selected parameters because they give enough countersteering (D) to keep the car stable at low speeds, but still provide aggressive enough proportional steering to keep the car on the track at high speeds.
