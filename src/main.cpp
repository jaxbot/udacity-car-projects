#include <fstream>
#include <math.h>
#include <uWS/uWS.h>
#include <chrono>
#include <iostream>
#include <thread>
#include <vector>
#include "Eigen-3.3/Eigen/Core"
#include "Eigen-3.3/Eigen/QR"
#include "json.hpp"
#include "spline.h"
#include "plannerstate.h"

using namespace std;

// for convenience
using json = nlohmann::json;

// For converting back and forth between radians and degrees.
constexpr double pi() { return M_PI; }
double deg2rad(double x) { return x * pi() / 180; }
double rad2deg(double x) { return x * 180 / pi(); }

// Checks if the SocketIO event has JSON data.
// If there is data the JSON object in string format will be returned,
// else the empty string "" will be returned.
string hasData(string s) {
  auto found_null = s.find("null");
  auto b1 = s.find_first_of("[");
  auto b2 = s.find_first_of("}");
  if (found_null != string::npos) {
    return "";
  } else if (b1 != string::npos && b2 != string::npos) {
    return s.substr(b1, b2 - b1 + 2);
  }
  return "";
}

int getLaneFromD(double car_d) {
  return car_d / 4;
}

double getLaneSpeed(int lane, double car_s, vector<vector<double>> sensor_fusion, double lookahead_distance = 30.0) {
  double min_s = 999999999;
  double speed = -1;
  for (int i = 0; i < sensor_fusion.size(); i++) {
    float d = sensor_fusion[i][6];

    if (d < (2 + 4 * lane + 2) && d > (2 + 4 * lane - 2)) {
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = sqrt(vx * vx + vy * vy);
      double check_car_s = sensor_fusion[i][5];

      if (check_car_s > car_s && check_car_s < min_s && check_car_s - car_s < lookahead_distance) {
	min_s = check_car_s;
	speed = check_speed;
      }
    }
  }

  return speed * 2.24; // return in mph
}

bool canMakeLaneChange(int lane, double car_s, vector<vector<double>> sensor_fusion) {
  double max_s = 0;
  double min_s = 999999999;
  double speed = -1;
  for (int i = 0; i < sensor_fusion.size(); i++) {
    float d = sensor_fusion[i][6];

    if (d < (2 + 4 * lane + 2) && d > (2 + 4 * lane - 2)) {
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_speed = sqrt(vx * vx + vy * vy);
      double check_car_s = sensor_fusion[i][5];

      // If it's behind us, find the closest car behind us
      if (check_car_s < (car_s + 10) && check_car_s > max_s) {
	max_s = check_car_s;
	speed = check_speed;
      }

      // Also keep track of the closest car in front of the desired lane so that we do not rear-end it
      if (check_car_s > car_s && check_car_s < min_s) {
	min_s = check_car_s;
      }
    }
  }

  return (car_s - max_s) > 15.0 && (min_s - car_s) > 15.0;
}

double distanceToLeadCar(int lane, double car_s, vector<vector<double>> sensor_fusion) {
  double min_s = 999999999;
  for (int i = 0; i < sensor_fusion.size(); i++) {
    float d = sensor_fusion[i][6];

    if (d < (2 + 4 * lane + 2) && d > (2 + 4 * lane - 2)) {
      double vx = sensor_fusion[i][3];
      double vy = sensor_fusion[i][4];
      double check_car_s = sensor_fusion[i][5];

      if (check_car_s > car_s && check_car_s < min_s) {
	min_s = check_car_s;
      }
    }
  }

  return min_s - car_s;
}

double distance(double x1, double y1, double x2, double y2)
{
	return sqrt((x2-x1)*(x2-x1)+(y2-y1)*(y2-y1));
}
int ClosestWaypoint(double x, double y, const vector<double> &maps_x, const vector<double> &maps_y)
{

	double closestLen = 100000; //large number
	int closestWaypoint = 0;

	for(int i = 0; i < maps_x.size(); i++)
	{
		double map_x = maps_x[i];
		double map_y = maps_y[i];
		double dist = distance(x,y,map_x,map_y);
		if(dist < closestLen)
		{
			closestLen = dist;
			closestWaypoint = i;
		}

	}

	return closestWaypoint;

}

int NextWaypoint(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{

	int closestWaypoint = ClosestWaypoint(x,y,maps_x,maps_y);

	double map_x = maps_x[closestWaypoint];
	double map_y = maps_y[closestWaypoint];

	double heading = atan2((map_y-y),(map_x-x));

	double angle = fabs(theta-heading);
  angle = min(2*pi() - angle, angle);

  if(angle > pi()/4)
  {
    closestWaypoint++;
  if (closestWaypoint == maps_x.size())
  {
    closestWaypoint = 0;
  }
  }

  return closestWaypoint;
}

// Transform from Cartesian x,y coordinates to Frenet s,d coordinates
vector<double> getFrenet(double x, double y, double theta, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int next_wp = NextWaypoint(x,y, theta, maps_x,maps_y);

	int prev_wp;
	prev_wp = next_wp-1;
	if(next_wp == 0)
	{
		prev_wp  = maps_x.size()-1;
	}

	double n_x = maps_x[next_wp]-maps_x[prev_wp];
	double n_y = maps_y[next_wp]-maps_y[prev_wp];
	double x_x = x - maps_x[prev_wp];
	double x_y = y - maps_y[prev_wp];

	// find the projection of x onto n
	double proj_norm = (x_x*n_x+x_y*n_y)/(n_x*n_x+n_y*n_y);
	double proj_x = proj_norm*n_x;
	double proj_y = proj_norm*n_y;

	double frenet_d = distance(x_x,x_y,proj_x,proj_y);

	//see if d value is positive or negative by comparing it to a center point

	double center_x = 1000-maps_x[prev_wp];
	double center_y = 2000-maps_y[prev_wp];
	double centerToPos = distance(center_x,center_y,x_x,x_y);
	double centerToRef = distance(center_x,center_y,proj_x,proj_y);

	if(centerToPos <= centerToRef)
	{
		frenet_d *= -1;
	}

	// calculate s value
	double frenet_s = 0;
	for(int i = 0; i < prev_wp; i++)
	{
		frenet_s += distance(maps_x[i],maps_y[i],maps_x[i+1],maps_y[i+1]);
	}

	frenet_s += distance(0,0,proj_x,proj_y);

	return {frenet_s,frenet_d};

}


// Transform from Frenet s,d coordinates to Cartesian x,y
vector<double> getXY(double s, double d, const vector<double> &maps_s, const vector<double> &maps_x, const vector<double> &maps_y)
{
	int prev_wp = -1;

	while(s > maps_s[prev_wp+1] && (prev_wp < (int)(maps_s.size()-1) ))
	{
		prev_wp++;
	}

	int wp2 = (prev_wp+1)%maps_x.size();

	double heading = atan2((maps_y[wp2]-maps_y[prev_wp]),(maps_x[wp2]-maps_x[prev_wp]));
	// the x,y,s along the segment
	double seg_s = (s-maps_s[prev_wp]);

	double seg_x = maps_x[prev_wp]+seg_s*cos(heading);
	double seg_y = maps_y[prev_wp]+seg_s*sin(heading);

	double perp_heading = heading-pi()/2;

	double x = seg_x + d*cos(perp_heading);
	double y = seg_y + d*sin(perp_heading);

	return {x,y};

}

int main() {
  uWS::Hub h;

  PlannerState state;
  state.state = Follow;
  state.lane = 1;
  state.desired_lane = 1;

  // Load up map values for waypoint's x,y,s and d normalized normal vectors
  vector<double> map_waypoints_x;
  vector<double> map_waypoints_y;
  vector<double> map_waypoints_s;
  vector<double> map_waypoints_dx;
  vector<double> map_waypoints_dy;

  // Waypoint map to read from
  string map_file_ = "../data/highway_map.csv";
  // The max s value before wrapping around the track back to 0
  double max_s = 6945.554;
  
  int atomic_timestep = 0;
  vector<double> velocity_at_timestep;

  ifstream in_map_(map_file_.c_str(), ifstream::in);

  string line;
  while (getline(in_map_, line)) {
  	istringstream iss(line);
  	double x;
  	double y;
  	float s;
  	float d_x;
  	float d_y;
  	iss >> x;
  	iss >> y;
  	iss >> s;
  	iss >> d_x;
  	iss >> d_y;
  	map_waypoints_x.push_back(x);
  	map_waypoints_y.push_back(y);
  	map_waypoints_s.push_back(s);
  	map_waypoints_dx.push_back(d_x);
  	map_waypoints_dy.push_back(d_y);
  }


  h.onMessage([&map_waypoints_x,&map_waypoints_y,&map_waypoints_s,&map_waypoints_dx,&map_waypoints_dy,&state,&atomic_timestep,&velocity_at_timestep](uWS::WebSocket<uWS::SERVER> ws, char *data, size_t length,
                     uWS::OpCode opCode) {
    // "42" at the start of the message means there's a websocket message event.
    // The 4 signifies a websocket message
    // The 2 signifies a websocket event
    //auto sdata = string(data).substr(0, length);
    //cout << sdata << endl;
    if (length && length > 2 && data[0] == '4' && data[1] == '2') {

      auto s = hasData(data);

      if (s != "") {
        auto j = json::parse(s);
        
        string event = j[0].get<string>();
        
        if (event == "telemetry") {
          // j[1] is the data JSON object
          
        	// Main car's localization Data
          	double car_x = j[1]["x"];
          	double car_y = j[1]["y"];
          	double car_s = j[1]["s"];
          	double car_d = j[1]["d"];
          	double car_yaw = j[1]["yaw"];
          	double car_speed = j[1]["speed"];

		state.speed = car_speed;

          	// Previous path data given to the Planner
          	auto previous_path_x = j[1]["previous_path_x"];
          	auto previous_path_y = j[1]["previous_path_y"];
          	// Previous path's end s and d values 
          	double end_path_s = j[1]["end_path_s"];
          	double end_path_d = j[1]["end_path_d"];

          	// Sensor Fusion Data, a list of all other cars on the same side of the road.
          	auto sensor_fusion = j[1]["sensor_fusion"];

          	json msgJson;

		double ref_x = car_x;
		double ref_y = car_y;
		double ref_yaw = deg2rad(car_yaw);
		double ref_velocity = 0;

          	vector<double> ptsx;
          	vector<double> ptsy;
		double reference_speed = 49.0;

		double dt = 0.02;

		if (previous_path_x.size() < 2) {
		  // Make a point tangent to the car.
		  double prev_car_x = car_x - cos(car_yaw);
		  double prev_car_y = car_y - sin(car_yaw);

		  ptsx.push_back(prev_car_x);
		  ptsx.push_back(car_x);

		  ptsy.push_back(prev_car_y);
		  ptsy.push_back(car_y);
		} else {
		  ref_x = previous_path_x[previous_path_x.size() - 1];
		  ref_y = previous_path_y[previous_path_y.size() - 1];

		  double ref_x_prev = previous_path_x[previous_path_x.size() - 2];
		  double ref_y_prev = previous_path_y[previous_path_y.size() - 2];
		  ref_yaw = atan2(ref_y - ref_y_prev, ref_x - ref_x_prev);

		  ptsx.push_back(ref_x_prev);
		  ptsx.push_back(ref_x);

		  ptsy.push_back(ref_y_prev);
		  ptsy.push_back(ref_y);

		  double dx = (ref_x - ref_x_prev) / dt;
		  double dy = (ref_y - ref_y_prev) / dt;
		  vector<double> frenet = getFrenet(ref_x, ref_y, ref_yaw, map_waypoints_x, map_waypoints_y);
		  vector<double> frenet_prev = getFrenet(ref_x_prev, ref_y_prev, ref_yaw, map_waypoints_x, map_waypoints_y);

		  ref_velocity = sqrt(dx*dx + dy*dy) * 2.24;//(frenet[1] - frenet_prev[1]) / dt * 2.24;
		  atomic_timestep += 50 - previous_path_x.size();
		}

		int lane = getLaneFromD(car_d);
		state.lane = lane;

		// figure out a safe lane speed
		double speed = getLaneSpeed(lane, car_s, sensor_fusion);
		double speed_far_out = getLaneSpeed(lane, car_s, sensor_fusion, 60.0);
		double following_distance = distanceToLeadCar(state.lane, car_s, sensor_fusion);
		std::cout << "Lane speed: " << speed << std::endl;
		if (speed > -1) {
		  reference_speed = speed;
		} else {
		  reference_speed = 49.0;
		  // If no car is within 30m, we can continue to accelerate.
		  // However, if a car exists within 60m, we should approach it at
		  // a safer speed. Otherwise we may come within 30m of a car going 30mph
		  // while we're going 50mph and have to slam on the brakes, causing jerk.
		  if (speed_far_out > -1 && following_distance > 31) {
		    reference_speed -= (reference_speed - speed_far_out) / 3.0;
		  }
		}

		state.lane_speed = reference_speed;

		// See if we can find a faster lane
		double fastest_speed = reference_speed;
		if (reference_speed < 49.0 && state.state != LaneChange) {
		  int fastest_lane = lane;

		  for (int i = 0; i < 3; i++) {
		    double lane_speed = getLaneSpeed(i, car_s, sensor_fusion);
		    if (lane_speed < 0) {
		      lane_speed = 49.0;
		    }
		    if (lane_speed > fastest_speed && abs(i - state.lane) <= 1) {
		      fastest_lane = i;
		      fastest_speed = lane_speed;
		    }
		  }

		  state.desired_lane = fastest_lane;
		}

		// update fastest speed when making a lane change to match the desired lane's speed
		if (state.state == LaneChange || state.state == PrepareLaneChange) {
		  fastest_speed = getLaneSpeed(state.desired_lane, car_s, sensor_fusion);
		  if (fastest_speed < 0) {
		    fastest_speed = 49.0;
		  }
		}

		if (state.lane != state.desired_lane) {
		  if (state.state == Follow) {
		    state.state = PrepareLaneChange;
		  }
		} else {
		  state.state = Follow;
		}

		if (state.state == PrepareLaneChange) {
		  if (canMakeLaneChange(state.desired_lane, car_s, sensor_fusion) && following_distance >= 15) {
		    // Do not make a lane change that would require significant braking at high speeds
		    if (ref_velocity - fastest_speed < 15 || ref_velocity < 42) {
		      state.state = LaneChange;
		    }
		  }
		}

		// Ensure we back off the car in front
		if (state.state == Follow || state.state == PrepareLaneChange) {
		  if (following_distance < 15) {
		    reference_speed -= 1.0;
		    cout << "Attempting to back off lead car " << following_distance << std::endl;
		    // Sometimes we have to brake to make a lane change. This can happen if we're going faster than our lane's normal speed, but our lane is still slower than an adjacent lane.
		  } else if (state.state == PrepareLaneChange && ref_velocity - fastest_speed >= 15 && ref_velocity >= 42) {
		    reference_speed -= 1.0;
		  }
		}

		// Find our next anchor points we plan to pass through.
		vector<vector<double>> anchor_points;
		int waypoint_spacing = 40.0;
		if (state.state == LaneChange) {
		  waypoint_spacing = 50.0;
		}
		for (int i = 1; i <= 3; i++) {
		  double d_path = (2 + 4 * lane);
		  double desired_lane_path = (2 + 4 * state.desired_lane);
		  if (state.state == LaneChange) {
		    d_path = (2 + 4 * state.desired_lane);
		    /*if (state.desired_lane < lane) { // left lane change
		      d_path = car_d - i * 1.5;
		    } else {
		      d_path = car_d + i * 1.5;
		    }*/
		    /*
		    cout << "D: " << d_path << " Desired: " << desired_lane_path << " Desired lane: " << state.desired_lane << std::endl;
		    if (car_d < 1 + 4 * lane)
		    if (i == 1) {
		      if (lane < state.desired_lane) {
			d_path = (4 * state.desired_lane);
		      } else {
			d_path = (4 * lane);
		      }
		    }
		    if (i == 3 || i == 2) {
		      d_path = (2 + 4 * state.desired_lane);
		    }
		    */
		    cout << "D: " << d_path << " Desired: " << desired_lane_path << std::endl;
		  }
		  vector<double> xy = getXY(car_s + waypoint_spacing * i, d_path, map_waypoints_s, map_waypoints_x, map_waypoints_y);
		  ptsx.push_back(xy[0]);
		  ptsy.push_back(xy[1]);
		}

		// Build a spline
		tk::spline spline_s;

		for (int i = 0; i < ptsx.size(); i++) {
		  double shift_x = ptsx[i] - ref_x;
		  double shift_y = ptsy[i] - ref_y;

		  ptsx[i] = (shift_x * cos(0 - ref_yaw) - shift_y * sin(0 - ref_yaw));
		  ptsy[i] = (shift_x * sin(0 - ref_yaw) + shift_y * cos(0 - ref_yaw));
		}

		spline_s.set_points(ptsx, ptsy);

          	vector<double> next_x_vals;
          	vector<double> next_y_vals;

		for (int i = 0; i < previous_path_x.size(); i++) {
		  next_x_vals.push_back(previous_path_x[i]);
		  next_y_vals.push_back(previous_path_y[i]);
		}

		double speed_limit = 49.0;
		double target_x = waypoint_spacing;
		double target_y = spline_s(target_x);
		double target_dist = sqrt((target_x * target_x) + (target_y * target_y));

		double x_add_on = 0;

		double velo_shift = 0.25;
		double current_speed = car_speed;

		// If we have points in the previous path, figure out the speed at the end of the path.
		if (state.state == LaneChange) {
		  reference_speed = fastest_speed;
		  // be more conservative about velocity shifts during lateral movements
		  velo_shift = 0.15;
		}
		if (previous_path_x.size() > 2) {
		  current_speed = ref_velocity + 0.1;
		  std::cout << "Current speed: " << current_speed << " vs " << car_speed << std::endl;
		}
		std::cout << "Reference speed " << reference_speed << std::endl;

		if (fabs(current_speed - reference_speed) < 1) {
		  current_speed = reference_speed;
		}

		if (atomic_timestep > 0) {
		  //current_speed = velocity_at_timestep[atomic_timestep];
		}

		if (state.state == LaneChange && reference_speed >= 49) {
		  reference_speed = 48; // slow a little during lane change because the simulator counts lateral movement itno the speed as well, which can cause speed violations
		}

		if (current_speed > speed_limit) {
		  current_speed = speed_limit;
		}

		for (int i = 1; i <= 50 - previous_path_x.size(); i++) {
		  if (current_speed > reference_speed) {
		    current_speed -= velo_shift;
		  } else if (current_speed < reference_speed) {
		    current_speed += velo_shift;
		  }

		  double N = (target_dist / ( 0.02 * current_speed / 2.24));
		  std::cout << "Generating " << target_dist << " points" << std::endl;
		  double x_point = x_add_on + target_x / N;
		  double y_point = spline_s(x_point);

		  x_add_on = x_point;

		  double x_ref = x_point;
		  double y_ref = y_point;

		  x_point = (x_ref * cos(ref_yaw) - y_ref * sin(ref_yaw));
		  y_point = (x_ref * sin(ref_yaw) + y_ref * cos(ref_yaw));

		  x_point += ref_x;
		  y_point += ref_y;

		  next_x_vals.push_back(x_point);
		  next_y_vals.push_back(y_point);

		  std::cout << "Loop current speed: " << current_speed << " vs " << car_speed << std::endl;

		  velocity_at_timestep.push_back(current_speed);
		}

          	msgJson["next_x"] = next_x_vals;
          	msgJson["next_y"] = next_y_vals;

          	auto msg = "42[\"control\","+ msgJson.dump()+"]";

          	//this_thread::sleep_for(chrono::milliseconds(1000));
          	ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
	} else if (event == "dashboard") {
	  // Send state payload to dashboard
	  json payload;
	  payload["speed"] = state.speed;
	  payload["lane_speed"] = state.lane_speed;
	  payload["lane"] = state.lane;
	  payload["state"] = state.state;
	  std::string msg = "42[\"dashboard\"," + payload.dump() + "]";
	  ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
	} else if (event == "lane_change") {
	  state.state = PrepareLaneChange;
	  state.desired_lane -= 1;
	  if (state.desired_lane < 0) {
	    state.desired_lane = 1;
	  }
	}
      } else {
        // Manual driving
        std::string msg = "42[\"manual\",{}]";
        ws.send(msg.data(), msg.length(), uWS::OpCode::TEXT);
      }
    }
  });

  // We don't need this since we're not using HTTP but if it's removed the
  // program
  // doesn't compile :-(
  h.onHttpRequest([](uWS::HttpResponse *res, uWS::HttpRequest req, char *data,
                     size_t, size_t) {
    const std::string s = "<h1>Hello world!</h1>";
    cout << req.getUrl().toString() << std::endl;
    std::string url = req.getUrl().toString();
    std::string filedata = "";
    std::string filename = "";
    std::string line = "";
    if (url == "/") {
      filename = "../src/dashboard.html";
    }
    if (url == "/dashboard.js") {
      filename = "../src/dashboard.js";
    }
    if (filename != "") {
      ifstream myfile (filename);
      if (myfile.is_open())
      {
      while ( getline (myfile,line) )
      {
	filedata += line;
      }
      myfile.close();
      }
      res->end(filedata.data(), filedata.length());
    } else {
      res->end(s.data(), s.length());
    }
  });

  h.onConnection([&h](uWS::WebSocket<uWS::SERVER> ws, uWS::HttpRequest req) {
    std::cout << "Connected!!!" << std::endl;
  });

  h.onDisconnection([&h](uWS::WebSocket<uWS::SERVER> ws, int code,
                         char *message, size_t length) {
    ws.close();
    std::cout << "Disconnected" << std::endl;
  });

  int port = 4567;
  if (h.listen(port)) {
    std::cout << "Listening to port " << port << std::endl;
  } else {
    std::cerr << "Failed to listen to port" << std::endl;
    return -1;
  }
  h.run();
}
