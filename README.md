# Multiplicative Extended Kalman Filter (MEKF)

C++ implementation of the Multiplicative Extended Kalman Filter (MEKF) 
for estimating an unmanned vehicle attitude using unit quaternion 
and position using latitude, longitude and altitude above WGS84 ellipsoid.
Supported sensors are accelerometer, magnetometer, barometer 
and GNSS position/attitude measurements (for dual GNSS antenna setup).

Based on the following article:
https://matthewhampsey.github.io/blog/2020/07/18/mekf

## Dependencies
- Eigen3 (libeigen3-dev)