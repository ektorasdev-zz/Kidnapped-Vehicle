/*
 * particle_filter.cpp
 *
 *  Created on: Dec 12, 2016
 *      Author: Tiffany Huang
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
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).

    num_particles = 100;

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    std_x     = std[0];
    std_y     = std[1];
    std_theta = std[2];

    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(x, std_x);
    normal_distribution<double> dist_y(y, std_y);
    normal_distribution<double> dist_theta(theta, std_theta);

    for (unsigned int i = 0; i < num_particles; ++i) {
        Particle part;
        part.id = i;
        part.x = dist_x(gen);
        part.y = dist_y(gen);
        part.theta = dist_theta(gen);
        part.weight = 1.0;

        particles.push_back(part);
    }
    is_initialized = true;
}

void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/

    default_random_engine gen;
    double std_x, std_y, std_theta; // Standard deviations for x, y, and theta

    std_x     = std_pos[0];
    std_y     = std_pos[1];
    std_theta = std_pos[2];

    // This line creates a normal (Gaussian) distribution for x
    normal_distribution<double> dist_x(0, std_x);
    normal_distribution<double> dist_y(0, std_y);
    normal_distribution<double> dist_theta(0, std_theta);

    for(unsigned int i = 0; i < num_particles; ++i) {
        if(fabs(yaw_rate) < 0.00001) {
            particles[i].x += (velocity * delta_t * particles[i].theta);
            particles[i].y += (velocity * delta_t * particles[i].theta);
        } else {
            particles[i].x     += (velocity/yaw_rate) * (sin(particles[i].theta + (yaw_rate * delta_t)) - sin(particles[i].theta));
            particles[i].y     += (velocity/yaw_rate) * (cos(particles[i].theta) - cos(particles[i].theta + (yaw_rate * delta_t)));
            particles[i].theta += yaw_rate * delta_t;
        }
        particles[i].x     += dist_x(gen);
        particles[i].y     += dist_y(gen);
        particles[i].theta += dist_theta(gen);
    }
}

void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.

    double min_dist = numeric_limits<double>::max();

    for(unsigned int i = 0; i < observations.size(); ++i) {
        LandmarkObs obs = observations[i];
        for(unsigned int j = 0; j < predicted.size(); ++j) {
            LandmarkObs prds = predicted[j];
            double curr_dist = dist(obs.x, obs.y, prds.x, prds.y);
            if(curr_dist < min_dist) {
                min_dist = curr_dist;
                obs.id = prds.id;
            }
        }
    }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[], 
		const std::vector<LandmarkObs> &observations, const Map &map_landmarks) {
    // TODO: Update the weights of each particle using a mult-variate Gaussian distribution. You can read
    //   more about this distribution here: https://en.wikipedia.org/wiki/Multivariate_normal_distribution
    // NOTE: The observations are given in the VEHICLE'S coordinate system. Your particles are located
    //   according to the MAP'S coordinate system. You will need to transform between the two systems.
    //   Keep in mind that this transformation requires both rotation AND translation (but no scaling).
    //   The following is a good resource for the theory:
    //   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
    //   and the following is a good resource for the actual equation to implement (look at equation
    //   3.33
    //   http://planning.cs.uiuc.edu/node99.html

    weights.clear();

    for(unsigned int i = 0; i < particles.size(); i++) {

        vector<LandmarkObs> map_obs;
        for(unsigned int j = 0; j < observations.size(); j++) {
            LandmarkObs obs;
            obs.x = observations[j].x * cos(particles[i].theta) - observations[j].y * sin(particles[i].theta) +
                    particles[i].x;
            obs.y = observations[j].x * sin(particles[i].theta) + observations[j].y * cos(particles[i].theta) +
                    particles[i].y;
            obs.id = observations[j].id;
            map_obs.push_back(obs);
        }

        vector<LandmarkObs> preds;
        double curr_dist;
        for(unsigned int l = 0; l < map_landmarks.landmark_list.size(); l++) {
            curr_dist = dist(particles[i].x, particles[i].y, map_landmarks.landmark_list[l].x_f,
                             map_landmarks.landmark_list[l].y_f);
            if (curr_dist < sensor_range) {
                LandmarkObs preds_landmark;
                preds_landmark.id = map_landmarks.landmark_list[l].id_i;
                preds_landmark.x = map_landmarks.landmark_list[l].x_f;
                preds_landmark.y = map_landmarks.landmark_list[l].y_f;
                preds.push_back(preds_landmark);
            }
        }

        dataAssociation(preds, map_obs);

        particles[i].weight = 1.0;
        for(unsigned int j=0; j<preds.size();++j) {

            bool flag = false;
            double mu_x, mu_y, dx, dy;
            double default_dist = 9;

            for(unsigned int k = 0; k < map_obs.size(); k++) {
                double min_dist = dist(preds[j].x, preds[j].y, map_obs[k].x, map_obs[k].y);
                if (min_dist < default_dist) {

                    default_dist = min_dist;
                    mu_x = preds[j].x;
                    mu_y = preds[j].y;

                    dx = mu_x - map_obs[k].x;
                    dy = mu_y - map_obs[k].y;

                    flag = true;
                }
            }
            if(flag) {
                particles[i].weight *= exp(-((dx * dx) / (2 * std_landmark[0]*std_landmark[0]) +
                              ((dy * dy) / (2 * std_landmark[1]*std_landmark[1]))))/
                        (2 * M_PI * std_landmark[0] * std_landmark[1]);
            }
        }
        weights.push_back(particles[i].weight);
    }
}

void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution

    vector<Particle> res_particles;
    default_random_engine gen;
    discrete_distribution<int>weight_dist(weights.begin(), weights.end());

    for(unsigned int i = 0; i < particles.size(); ++i) {
        res_particles.push_back(particles[weight_dist(gen)]);
    }
    particles = res_particles;
}

Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

    particle.associations.clear();
    particle.sense_x.clear();
    particle.sense_y.clear();

    particle.associations = associations;
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
