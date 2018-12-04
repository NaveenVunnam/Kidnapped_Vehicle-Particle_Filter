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
static default_random_engine gen;
void ParticleFilter::init(double x, double y, double theta, double std[]) {
	// TODO: Set the number of particles. Initialize all particles to first position (based on estimates of 
	//   x, y, theta and their uncertainties from GPS) and all weights to 1. 
	// Add random Gaussian noise to each particle.
	// NOTE: Consult particle_filter.h for more information about this method (and others in this file).
  
  num_particles = 100;
  double std_x = std[0];
  double std_y = std[1];
  double std_theta = std[2];  
  normal_distribution<double> dist_x(x,std_x);
  normal_distribution<double> dist_y(y,std_y);
  normal_distribution<double> dist_theta(theta,std_theta);  
  for (int i=0; i < num_particles; i++) {
    Particle P;
    P.id = i;
    P.x = dist_x(gen);
    P.y = dist_y(gen);
    P.theta = dist_theta(gen);
    P.weight = 1.0;
    particles.push_back(P);
  }
  is_initialized = true;  
}
void ParticleFilter::prediction(double delta_t, double std_pos[], double velocity, double yaw_rate) {
	// TODO: Add measurements to each particle and add random Gaussian noise.
	// NOTE: When adding noise you may find std::normal_distribution and std::default_random_engine useful.
	//  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
	//  http://www.cplusplus.com/reference/random/default_random_engine/ 
  double std_x = std_pos[0];
  double std_y = std_pos[1];
  double std_theta = std_pos[2];
  normal_distribution<double> dist_x(0,std_x);
  normal_distribution<double> dist_y(0,std_y);
  normal_distribution<double> dist_theta(0,std_theta);  
  for (int i=0; i < num_particles; i++) {
    double yaw = particles[i].theta;    
    if (fabs(yaw_rate) > 0.001) {
      particles[i].x += velocity/yaw_rate * (sin(yaw + yaw_rate * delta_t) - sin(yaw));
      particles[i].y += velocity/yaw_rate * (cos(yaw) - cos(yaw + yaw_rate*delta_t));
      particles[i].theta += yaw_rate * delta_t;
    } else {
      particles[i].x += velocity * delta_t * cos(yaw);
      particles[i].y += velocity * delta_t * sin(yaw);
    }
    // Add Gaussian Noise
    particles[i].x += dist_x(gen);
    particles[i].y += dist_y(gen);
    particles[i].theta += dist_theta(gen);
  }
}
void ParticleFilter::dataAssociation(std::vector<LandmarkObs> predicted, std::vector<LandmarkObs>& observations) {
	// TODO: Find the predicted measurement that is closest to each observed measurement and assign the 
	//   observed measurement to this particular landmark.
	// NOTE: this method will NOT be called by the grading code. But you will probably find it useful to 
	//   implement this method and use it as a helper during the updateWeights phase.
  for (unsigned int i=0; i < observations.size(); i++) {
    LandmarkObs obvs = observations[i];
    double min_dist = numeric_limits<double>::max();
    int num_id = -1;
    for (unsigned int j=0; j < predicted.size(); j++) {
      LandmarkObs preds = predicted[j];
      double distance = sqrt((pow((obvs.x-preds.x),2)) + (pow((obvs.y-preds.y),2)));
      if (distance < min_dist) {
        min_dist = distance;
        num_id = preds.id;
      }
    }
    observations[i].id = num_id;
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
  for (int i=0; i < num_particles; i++) {
    double p_x = particles[i].x;
    double p_y = particles[i].y;
    double p_theta = particles[i].theta;
     // Translate into Map coordinates
    vector<LandmarkObs> trans_obs;
    for (unsigned int j=0; j < observations.size(); j++) {
      double t_x = p_x + (cos(p_theta) * observations[j].x) - (sin(p_theta) * observations[j].y);
      double t_y = p_y + (sin(p_theta) * observations[j].x) + (cos(p_theta) * observations[j].y);
      trans_obs.push_back(LandmarkObs{observations[j].id, t_x, t_y});
    }    
    // Find Landmark predictions within the range of the Sensor
    vector<LandmarkObs> Predictions;
    for (unsigned int j=0; j < map_landmarks.landmark_list.size(); j++) {
      float lm_x = map_landmarks.landmark_list[j].x_f;
      float lm_y = map_landmarks.landmark_list[j].y_f;
      int lm_id = map_landmarks.landmark_list[j].id_i;
      if ((fabs(lm_x - p_x) <= sensor_range) && (fabs(lm_y - p_y) <= sensor_range)) {
        Predictions.push_back(LandmarkObs{lm_id, lm_x, lm_y});
      }
    }   
    // Data association Step
    dataAssociation(Predictions,trans_obs);
    // Update weights
    particles[i].weight = 1.0; // re-initialize weights
    for (unsigned int k=0; k < trans_obs.size(); k++) {
      double obv_x, obv_y, pred_x, pred_y;
      obv_x = trans_obs[k].x;
      obv_y = trans_obs[k].y;
      int obv_id = trans_obs[k].id;
      for (unsigned int l=0; l < Predictions.size(); l++) {
        if (obv_id == Predictions[l].id) {
          pred_x = Predictions[l].x;
          pred_y = Predictions[l].y;
        }
      }
      double sd_x = std_landmark[0];
      double sd_y = std_landmark[1];
      double obv_w = ( 1/(2*M_PI*sd_x*sd_y)) * exp( -( pow(pred_x-obv_x,2)/(2*pow(sd_x, 2)) + (pow(pred_y-obv_y,2)/(2*pow(sd_y, 2))) ) );
      particles[i].weight *= obv_w;
    }
  }
}
void ParticleFilter::resample() {
	// TODO: Resample particles with replacement with probability proportional to their weight. 
	// NOTE: You may find std::discrete_distribution helpful here.
	//   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
  vector<double> weights;
  vector<Particle> new_particles;  
  for (int i=0; i < num_particles; i++) {
    weights.push_back(particles[i].weight);    
  }
  double max_w = *max_element(weights.begin(), weights.end());
  uniform_real_distribution<double> distDouble(0.0, max_w);
  uniform_int_distribution<int> distInt(0, num_particles-1);
  int index = distInt(gen);
  double beta = 0.0;
  for (int i=0; i < num_particles; i++) {
    beta += distDouble(gen) * 2.0 * max_w;
    while (beta > weights[index]) {
      beta -= weights[index];
      index = (index + 1) % num_particles;
    }
    new_particles.push_back(particles[index]);
  }
  particles = new_particles;
}
Particle ParticleFilter::SetAssociations(Particle& particle, const std::vector<int>& associations, 
                                     const std::vector<double>& sense_x, const std::vector<double>& sense_y)
{
    //particle: the particle to assign each listed association, and association's (x,y) world coordinates mapping to
    // associations: The landmark id that goes along with each listed association
    // sense_x: the associations x mapping already converted to world coordinates
    // sense_y: the associations y mapping already converted to world coordinates

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
