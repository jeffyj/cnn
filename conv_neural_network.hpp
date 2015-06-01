/* machine-learning-lib
 * convolutional-neural-network (cnn)
 * 
 * Copyright (c) 2012 J.J.
 *
 * This program is free software: you can redistribute it and/or
 * modify it under the terms of the GNU General Public License as
 * published by the Free Software Foundation, either version 3 of the
 * License, or (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 *
 * $Id: conv_neural_network.hpp 7 2012-12-16 03:33:24Z pondfiller1@gmail.com $
 */

#ifndef CONV_NEURAL_NETWORK_HPP
#define CONV_NEURAL_NETWORK_HPP

#include <boost/cstdint.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include <boost/shared_ptr.hpp>
#include <boost/shared_array.hpp>
#include <boost/enable_shared_from_this.hpp>

namespace conv_neural_network {

/* basic typedefs */
typedef boost::uint32_t uint32_t;
typedef boost::uint16_t uint16_t;
typedef boost::uint8_t uint8_t;

#define SIGMOID(x) (1.7159 * tanh(0.66666667 * x)) 
#define DSIGMOID(S) (0.66666667 / 1.7159 * (1.7159 + (S)) * (1.7159 - (S)))
#define UNIFORM_PLUS_MINUS_ONE ((double)(2.0 * rand())/RAND_MAX - 1.0)

typedef enum
{
  INPUT_LAYER,
  CONV_LAYER,
  OUTPUT_LAYER
} layer_t;

typedef enum 
{ 
  FIRST_ORDER_DV, 
  SECOND_ORDER_DV 
} dv_order_t;

/* feature_size = ((29 - 5) / sampling_factor) + 1
/**
 * feature_map
 */
struct layer;

struct feature_map
{
  feature_map();

  void initialize();

  void clear();
  void clear_values();
  void clear_kernel();
  void clear_derror();

  void calculate();
  double convolve(const feature_map& feature_map, uint32_t feature_size, 
                  const boost::numeric::ublas::matrix<double>& kernel, uint32_t kernel_size,
                  uint32_t row, uint32_t column);

  boost::shared_ptr<layer> parent_layer_;
  boost::shared_array<boost::numeric::ublas::matrix<double> > kernels_;
  boost::shared_array<boost::numeric::ublas::matrix<double> > diag_hessians_;
  boost::shared_array<boost::numeric::ublas::matrix<double> > delta_w_ij_;
  /* y = f(x) = output after sigmoid is applied */
  boost::numeric::ublas::vector<double> y_;
  /* x = input to neuron */
  boost::numeric::ublas::vector<double> x_; 
  boost::numeric::ublas::vector<double> derror_;
  uint32_t num_feature_maps_prev_layer_;
  double bias_;
};

/**
 * layer
 */
struct layer : public boost::enable_shared_from_this<layer>
{
  layer(layer_t type, uint32_t num_feature_maps, uint32_t feature_size, 
        uint32_t kernel_size, uint32_t sampling_factor);

  void initialize();
  void initialize_feature_maps();
  void clear_all_feature_map_values();

  void forward_propagate();
  void backward_propagate(dv_order_t order, double learning_rate);

  boost::shared_array<feature_map> feature_maps_;
  boost::shared_ptr<layer> prev_layer_;

  /* layer parameters */
  layer_t type_;
  uint32_t num_feature_maps_;
  uint32_t feature_size_;
  uint32_t kernel_size_;
  uint32_t sampling_factor_;
};

/**
 * neural network 
 */
class neural_network
{
public:
  void add_layer(layer_t type, uint32_t num_feature_maps, 
                 uint32_t feature_size, uint32_t kernel_size,
                 uint32_t sampling_factor);

  int forward_propagate(const boost::numeric::ublas::vector<double>& input,
                        boost::numeric::ublas::vector<double>& output);
  void backward_propagate(const boost::numeric::ublas::vector<double> desired_output, double learning_rate);

private:
  std::vector<boost::shared_ptr<layer> > layers_;
};







} // namespace conv_neural_network

#endif /* CONV_NEURAL_NETWORK_HPP */