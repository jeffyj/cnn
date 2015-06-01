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
 * $Id: forward_propagation.cpp 7 2012-12-16 03:33:24Z pondfiller1@gmail.com $
 */

#include <conv_neural_network.hpp>

/* cnn */
namespace conv_neural_network {

int neural_network::forward_propagate(const boost::numeric::ublas::vector<double>& input,
                                      boost::numeric::ublas::vector<double>& output)
{
  BOOST_ASSERT(layers_.size() != 0);
  BOOST_ASSERT(layers_[INPUT_LAYER]->type_ == INPUT_LAYER);
  BOOST_ASSERT(layers_[INPUT_LAYER]->num_feature_maps_ == input.size());
  uint32_t i;

  /* initialize values for input layer feature map, 29x29 image */
  for(i = 0; i < (layers_[0]->feature_size_ * layers_[0]->feature_size_); i++)
  {
    layers_[INPUT_LAYER]->feature_maps_[0].x_[i] = input[i];
    layers_[INPUT_LAYER]->feature_maps_[0].y_[i] = input[i];
  }

  /* forward propagate all the remaining layers */
  for(i = 1; i < layers_.size(); i++)
  {
    layers_[i]->clear_all_feature_map_values();
    layers_[i]->forward_propagate();
  }
}

void layer::forward_propagate()
{
  for(uint32_t i = 0; i < num_feature_maps_; i++)
  {
    feature_maps_[i].calculate();
  }
}

void feature_map::calculate()
{
  boost::shared_ptr<layer> prev_layer = parent_layer_->prev_layer_;
  uint32_t feature_size = prev_layer->feature_size_;
  uint32_t kernel_size = prev_layer->kernel_size_;
  uint32_t diff = feature_size - kernel_size;
  uint32_t j = 0;

  /* convolve each feature map in the previous layer with their
     corresponding kernels and add the results to the bias to
     obtain the values in the current feature map */
  for(uint32_t i = 0; i < num_feature_maps_prev_layer_; i++)
  {
    for(uint32_t row = 0; row <= diff; row++)
    {
      for(uint32_t column = 0; column <= diff; row++)
      {
        x_[j] = bias_ + convolve(prev_layer->feature_maps_[i], feature_size, 
          kernels_[i], kernel_size, row, column);
        y_[j++] = SIGMOID(x_[j]);
      }
    }
  }
}

double feature_map::convolve(const feature_map& feature_map, uint32_t feature_size, 
                             const boost::numeric::ublas::matrix<double>& kernel, uint32_t kernel_size,
                             uint32_t row, uint32_t column)
{
  double sum = 0;
  
  /* convolve a matrix of feature_size x feature_size with a 
     kernel of kernel_size x kernel_size, beginning at 
     (row, column) */
  for(uint32_t i = row; i < (row + kernel_size); i++)
  {
    for(uint32_t j = column; j < (column + kernel_size); j++)
    {
      sum += (feature_map.y_[i * feature_size + j] * kernel(i, j));
    }
  }

  return sum;
}

} // namespace conv_neural_network