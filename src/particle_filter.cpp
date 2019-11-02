/**
 * particle_filter.cpp
 *
 * Created on: Dec 12, 2016
 * Author: Tiffany Huang
 */

#include "particle_filter.h"

#include <math.h>
#include <algorithm>
#include <iostream>
#include <iterator>
#include <numeric>
#include <random>
#include <string>
#include <vector>

#include "helper_functions.h"

using std::string;
using std::vector;
using std::normal_distribution;
using std::sin;
using std::cos;
using std::sqrt;
using std::pow;
using std::numeric_limits;
using std::exp;

void ParticleFilter::init(double x, double y, double theta, double std[]) {
  /**
   * TODO: Set the number of particles. Initialize all particles to
   *   first position (based on estimates of x, y, theta and their uncertainties
   *   from GPS) and all weights to 1.
   * TODO: Add random Gaussian noise to each particle.
   * NOTE: Consult particle_filter.h for more information about this method
   *   (and others in this file).
   */
  num_particles = 100;  // TODO: Set the number of particles
  std::default_random_engine gen;
  normal_distribution<double> dist_x(x, std[0]);
  normal_distribution<double> dist_y(y, std[1]);
  normal_distribution<double> dist_theta(theta, std[2]);
  particles.clear();
  for (int i = 0; i < num_particles; i ++) {
    Particle *particle = new Particle();
    particle->x = dist_x(gen);
    particle->y = dist_y(gen);
    particle->theta = dist_theta(gen);
    particle->weight = 1;
    particles.push_back(*particle);
  }
}

void ParticleFilter::prediction(double delta_t, double std_pos[],
                                double velocity, double yaw_rate) {
  /**
   * TODO: Add measurements to each particle and add random Gaussian noise.
   * NOTE: When adding noise you may find std::normal_distribution
   *   and std::default_random_engine useful.
   *  http://en.cppreference.com/w/cpp/numeric/random/normal_distribution
   *  http://www.cplusplus.com/reference/random/default_random_engine/
   */
  std::default_random_engine gen;
  normal_distribution<double> dist_velocity(velocity, std_pos[0]);
  normal_distribution<double> dist_yaw_rate(yaw_rate, std_pos[1]);
  for (int i = 0; i < num_particles; i ++) {
    double v = dist_velocity(gen);
    double y_r = dist_yaw_rate(gen);
    double theta = particles[i].theta;
    particles[i].x += (v / yaw_rate)*(sin(theta+y_r*delta_t)-sin(theta));
    particles[i].y +=  (v / yaw_rate)*(cos(theta) - cos(theta + y_r*delta_t));
    particles[i].theta += y_r*delta_t;
  }
}

void ParticleFilter::dataAssociation(vector<LandmarkObs> predicted,
                                     vector<LandmarkObs>& observations) {
  /**
   * TODO: Find the predicted measurement that is closest to each
   *   observed measurement and assign the observed measurement to this
   *   particular landmark.
   * NOTE: this method will NOT be called by the grading code. But you will
   *   probably find it useful to implement this method and use it as a helper
   *   during the updateWeights phase.
   */
  for (vector<LandmarkObs>::iterator obs = observations.begin();
      obs != observations.end(); ++obs) {
    double min_len = numeric_limits<double>::max();
    for (vector<LandmarkObs>::iterator pred = predicted.begin();
        pred != predicted.end(); ++pred) {
      double len = dist(obs->x, obs->y, pred->x, pred->y);
      if (len < min_len) {
        obs->id = pred->id;
        min_len = len;
      }
    }
  }
}

void ParticleFilter::updateWeights(double sensor_range, double std_landmark[],
                                   const vector<LandmarkObs> &observations,
                                   const Map &map_landmarks) {
  /**
   * TODO: Update the weights of each particle using a mult-variate Gaussian
   *   distribution. You can read more about this distribution here:
   *   https://en.wikipedia.org/wiki/Multivariate_normal_distribution
   * NOTE: The observations are given in the VEHICLE'S coordinate system.
   *   Your particles are located according to the MAP'S coordinate system.
   *   You will need to transform between the two systems. Keep in mind that
   *   this transformation requires both rotation AND translation (but no scaling).
   *   The following is a good resource for the theory:
   *   https://www.willamette.edu/~gorr/classes/GeneralGraphics/Transforms/transforms2d.htm
   *   and the following is a good resource for the actual equation to implement
   *   (look at equation 3.33) http://planning.cs.uiuc.edu/node99.html
   */
  for (int i=0; i < num_particles; i++) {
    vector<LandmarkObs> o;
    for (auto obs = observations.begin() ; obs != observations.end(); ++obs) {
      LandmarkObs *lo = new LandmarkObs();
      lo->x=particles[i].x+obs->x*cos(particles[i].theta)-
          obs->y*sin(particles[i].theta);
      lo->y=particles[i].y+obs->x*sin(particles[i].theta)+
          obs->y*cos(particles[i].theta);
      lo->id=obs->id;
      o.push_back(*lo);
    }
    vector<LandmarkObs> l;
    for (auto lm = map_landmarks.landmark_list.begin();
        lm != map_landmarks.landmark_list.end(); ++lm) {
      LandmarkObs *lo = new LandmarkObs();
      if (dist(particles[i].x, particles[i].y, (double)lm->x_f,
               (double)lm->y_f) <= sensor_range) {
        lo->x = (double)lm->x_f;
        lo->y = (double)lm->y_f;
        lo->id = lm->id_i;
        l.push_back(*lo);
      }
    }
    dataAssociation(l,o);
    for (vector<LandmarkObs>::iterator obs = o.begin(); obs != o.end(); ++obs) {
      for (vector<LandmarkObs>::iterator lm = l.begin(); lm != l.end(); ++lm) {
        if (lm->id == obs->id) {
          particles[i].weight *= (1.0/(2*M_PI*std_landmark[0]*std_landmark[1]))*
              exp(-(pow(obs->x-lm->x,2)/(2*pow(std_landmark[0],2))+
                  pow(obs->y-lm->y,2)/(2*pow(std_landmark[1],2))));
        }
      }
    }
  }
}

void ParticleFilter::resample() {
  /**
   * TODO: Resample particles with replacement with probability proportional
   *   to their weight.
   * NOTE: You may find std::discrete_distribution helpful here.
   *   http://en.cppreference.com/w/cpp/numeric/random/discrete_distribution
   */
  for (int i = 0; i < num_particles; i++) {
    weights.clear();
    weights.push_back(particles[i].weight);
  }
  std::random_device rd;
  std::mt19937 gen(rd());
  std::discrete_distribution<> d(weights.begin(), weights.end());
  std::vector<Particle> pl;
  for (int i = 0; i < num_particles; i++) {
    Particle *p = new Particle();
    *p = particles[d(gen)];
    pl.push_back(*p);
  }
  for (int i = 0; i < num_particles; i++) {
    particles[i] = pl[i];
  }
}

void ParticleFilter::SetAssociations(Particle& particle,
                                     const vector<int>& associations,
                                     const vector<double>& sense_x,
                                     const vector<double>& sense_y) {
  // particle: the particle to which assign each listed association,
  //   and association's (x,y) world coordinates mapping
  // associations: The landmark id that goes along with each listed association
  // sense_x: the associations x mapping already converted to world coordinates
  // sense_y: the associations y mapping already converted to world coordinates
  particle.associations= associations;
  particle.sense_x = sense_x;
  particle.sense_y = sense_y;
}

string ParticleFilter::getAssociations(Particle best) {
  vector<int> v = best.associations;
  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<int>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}

string ParticleFilter::getSenseCoord(Particle best, string coord) {
  vector<double> v;

  if (coord == "X") {
    v = best.sense_x;
  } else {
    v = best.sense_y;
  }

  std::stringstream ss;
  copy(v.begin(), v.end(), std::ostream_iterator<float>(ss, " "));
  string s = ss.str();
  s = s.substr(0, s.length()-1);  // get rid of the trailing space
  return s;
}
