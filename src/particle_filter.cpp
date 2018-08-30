/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
 *
 *  Modified by Jonathan Warner for CarND Project 3
 */

#include <random>
#include <algorithm>
#include <iostream>
#include <numeric>
#include <math.h> 
#include <iostream>
#include <sstream>
#include <string>
#include <iterator>

#include "particle_filter.h"

using namespace std;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
    default_random_engine gen;

    num_particles = 100;
    is_initialized = true;

    normal_distribution<double> dist_x(0, std[0]);
    normal_distribution<double> dist_y(0, std[1]);
    normal_distribution<double> dist_theta(0, std[2]);

    // Initialize each particle and add to particles vector.
    for (int i = 0; i < num_particles; i++) {
        Particle particle;
        // Particle location is provided coarse location + random noise centered at 0.
        particle.x = x + dist_x(gen);
        particle.y = y + dist_y(gen);
        particle.theta = theta + dist_theta(gen);
        particle.weight = 1.0;
        particle.id = i;

        particles.push_back(particle);
    }
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
    default_random_engine gen;

    normal_distribution<double> dist_x(0, std_pos[0]);
    normal_distribution<double> dist_y(0, std_pos[1]);
    normal_distribution<double> dist_theta(0, std_pos[2]);

    // Prediction phase. Predict the x, y, theta of each particle and add noise.
    for (int i = 0; i < num_particles; i++) {
        Particle p = particles[i];

        // If yaw_rate != ~0, use trig formula.
        if (fabs(yaw_rate) > 0.00001) {
            particles[i].x += (velocity / yaw_rate) * (sin(p.theta + yaw_rate * delta_t) - sin(p.theta)) + dist_x(gen);
            particles[i].y += (velocity / yaw_rate) * (cos(p.theta) - cos(p.theta + yaw_rate * delta_t)) + dist_y(gen);
            particles[i].theta += dist_theta(gen) + delta_t * yaw_rate;
        } else {
            // Otherwise assume constant yaw
            particles[i].x += (velocity * delta_t) * cos(p.theta) + dist_x(gen);
            particles[i].y += (velocity * delta_t) * sin(p.theta) + dist_y(gen);
            particles[i].theta += dist_theta(gen);
        }
    }

}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
    // Match each observation with its closest prediction by euclidean distance.
    for (int i = 0; i < observations.size(); i++) {
        int current_id = 0;
        double lowest_distance = std::numeric_limits<double>::max();
        for (int j = 0; j < predicted.size(); j++) {
            double distance = dist(
                    observations[i].x, observations[i].y,
                    predicted[j].x, predicted[j].y);

            if (distance < lowest_distance) {
                lowest_distance = distance;
                current_id = predicted[j].id;
            }
        }

        observations[i].id = current_id;
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
        const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // NOTE: The observations are given in the VEHICLE'S coordinate system.
    // Your particles are located according to the MAP'S coordinate system.
    // You will need to transform between the two systems.

    // Update weight of each particle.
    weights.clear();

    for (int i = 0; i < num_particles; i++) {
        Particle p = particles[i];

        std::vector<LandmarkObs> transformed_observations;
        std::vector<LandmarkObs> viable_landmarks;

        // Convert observations from vehicle coordinates to map coordinates.
        for (int j = 0; j < observations.size(); j++) {
            LandmarkObs ob = observations[j];

            double map_x = p.x + ob.x * cos(p.theta) - ob.y * sin(p.theta);
            double map_y = p.y + ob.y * cos(p.theta) + ob.x * sin(p.theta);

            LandmarkObs transformed_ob = {
                ob.id,
                map_x,
                map_y
            };

            transformed_observations.push_back(transformed_ob);
        }

        // Create a list of landmarks within the sensors range.
        for (int j = 0; j < map_landmarks.landmark_list.size(); j++) {
            auto landmark = map_landmarks.landmark_list[j];
            double distance = dist(landmark.x_f, landmark.y_f, p.x, p.y);

            if (distance <= sensor_range) {
                LandmarkObs observed_landmark = {
                    landmark.id_i,
                    landmark.x_f,
                    landmark.y_f
                };
                viable_landmarks.push_back(observed_landmark);
            }
        }

        // Associate landmarks and observations using nearest-neighbor.
        this->dataAssociation(viable_landmarks, transformed_observations);

        // Store associations between sensor readings and the associated landmark ID for
        // displaying in similar.
        vector<int> associated_id;
        vector<double> sense_x;
        vector<double> sense_y;

        // Calculate new particle weight.
        double weight = 1.0;

        for (int j = 0; j < transformed_observations.size(); j++) {
            LandmarkObs ob = transformed_observations[j];

            double p_x, p_y;

            // Find ground-truth landmark location associated with this observation via
            // ParticleFilter->dataAssociation call above.
            for (int n = 0; n < viable_landmarks.size(); n++) {
                if (viable_landmarks[n].id == ob.id) {
                    p_x = viable_landmarks[n].x;
                    p_y = viable_landmarks[n].y;
                    break;
                }
            }

            associated_id.push_back(ob.id);
            sense_x.push_back(ob.x);
            sense_y.push_back(ob.y);

            // Calculate new weight scaler using equation from module.
            double c = 1 / (2 * M_PI * std_landmark[0] * std_landmark[1]);
            double xv = (pow(p_x - ob.x, 2) / (2 * pow(std_landmark[0], 2)));
            double yv = (pow(p_y - ob.y, 2) / (2 * pow(std_landmark[1], 2)));
            weight *= c * exp(-(xv + yv));
        }

        particles[i].weight = weight;
        this->weights.push_back(weight);

        particles[i] = SetAssociations(particles[i], associated_id, sense_x, sense_y);
    }
}

void ParticleFilter::resample() {
    // Resample particles with resampling probability proportional to their weights.
    default_random_engine gen;

    vector<Particle> resampled_particles;
    discrete_distribution<int> dist(weights.begin(), weights.end());

    for (int i = 0; i < num_particles; i++) {
        int index = dist(gen);
        resampled_particles.push_back(particles[index]);
    }

    particles = resampled_particles;
}

Particle ParticleFilter::SetAssociations(Particle particle, std::vector<int> associations, std::vector<double> sense_x, std::vector<double> sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    //Clear the previous associations
    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations= associations;
    particle.sense_x = sense_x;
    particle.sense_y = sense_y;

    return particle;
}

string ParticleFilter::getAssociations(Particle best)
{
    vector<int> v = best.associations;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<int>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseX(Particle best)
{
    vector<double> v = best.sense_x;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
string ParticleFilter::getSenseY(Particle best)
{
    vector<double> v = best.sense_y;
    stringstream ss;
    copy( v.begin(), v.end(), ostream_iterator<float>(ss, " "));
    string s = ss.str();
    s = s.substr(0, s.length()-1);  // get rid of the trailing space
    return s;
}
