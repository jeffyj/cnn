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
 * $Id: conv_neural_network.cpp 7 2012-12-16 03:33:24Z pondfiller1@gmail.com $
 */

#include <conv_neural_network.hpp>

/* cnn */
namespace conv_neural_network {

/** 
 * feature_map 
 */
feature_map::feature_map() : num_feature_maps_prev_layer_(0)
{
}

void feature_map::initialize()
{
  if (parent_layer_->type_ != INPUT_LAYER)
  {
    num_feature_maps_prev_layer_ = parent_layer_->prev_layer_->num_feature_maps_;
  }

  uint32_t kernel_size = parent_layer_->kernel_size_;
  uint32_t feature_size = parent_layer_->feature_size_;

  x_.resize(feature_size * feature_size);
  y_.resize(feature_size * feature_size);

  derror_.resize(feature_size * feature_size);

  /* all the kernels from the previous layer that contribute to this feature map */
  kernels_ = boost::shared_array<boost::numeric::ublas::matrix<double> >(
    new boost::numeric::ublas::matrix<double> [num_feature_maps_prev_layer_]
  );

  diag_hessians_ = boost::shared_array<boost::numeric::ublas::matrix<double> >(
    new boost::numeric::ublas::matrix<double> [num_feature_maps_prev_layer_]
  );

  delta_w_ij_ = boost::shared_array<boost::numeric::ublas::matrix<double> >(
    new boost::numeric::ublas::matrix<double> [num_feature_maps_prev_layer_]
  );

  for(uint32_t i = 0; i < num_feature_maps_prev_layer_; i++)
  {
    /* set matrix sizes to kernel size KxK */
    kernels_[i].resize(kernel_size, kernel_size);
    diag_hessians_[i].resize(kernel_size, kernel_size);
    delta_w_ij_[i].resize(kernel_size, kernel_size);

    /* initialize kernel weights + bias to random */
    bias_ = 0.05 * UNIFORM_PLUS_MINUS_ONE;
    
    for(uint32_t j = 0; j < kernel_size; j++)
      for(uint32_t k = 0; k < kernel_size; k++)
        kernels_[i](j, k) = 0.05 * UNIFORM_PLUS_MINUS_ONE;
  }
}

void feature_map::clear()
{
  clear_values();
  clear_derror();
}

void feature_map::clear_values()
{
  x_.clear();
  y_.clear();
}

void feature_map::clear_derror()
{
  derror_.clear();
}

/**
 * layer
 */
layer::layer(layer_t type, uint32_t num_feature_maps, uint32_t feature_size, 
             uint32_t kernel_size, uint32_t sampling_factor) 
             : type_(type), 
               num_feature_maps_(num_feature_maps),
               feature_size_(feature_size),
               kernel_size_(kernel_size),
               sampling_factor_(sampling_factor)
{
  feature_maps_ = boost::shared_array<feature_map>(new feature_map [num_feature_maps]);
}

void layer::initialize()
{
  initialize_feature_maps();
}

void layer::initialize_feature_maps()
{
  for(uint32_t i = 0; i < num_feature_maps_; i++)
  {
    feature_maps_[i].parent_layer_ = shared_from_this();
    feature_maps_[i].initialize();
  }
}

void layer::clear_all_feature_map_values()
{
  for(uint32_t i = 0; i < num_feature_maps_; i++)
  {
    feature_maps_[i].clear_values();
  }
}

/**
 * neural network 
 */
void neural_network::add_layer(layer_t type, uint32_t num_feature_maps, 
                               uint32_t feature_size, uint32_t kernel_size,
                               uint32_t sampling_factor)
{
  boost::shared_ptr<layer> new_layer(
    new layer(type, num_feature_maps, feature_size, kernel_size, sampling_factor)
  );

  new_layer->initialize();
}

} // namespace conv_neural_network
