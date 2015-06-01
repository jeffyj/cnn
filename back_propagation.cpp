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
 * $Id: back_propagation.cpp 9 2012-12-16 03:41:23Z pondfiller1@gmail.com $
 */

#include <conv_neural_network.hpp>

/* cnn */
namespace conv_neural_network {

void neural_network::backward_propagate(const boost::numeric::ublas::vector<double> desired_output, 
                                        double learning_rate)
{
  /* input layer + hidden layer + output layer == minimum */
  BOOST_ASSERT(layers_.size() > 2);

  boost::shared_ptr<layer> output_layer = layers_.back();
  boost::shared_ptr<layer> prev_layer = output_layer->prev_layer_;
  uint32_t i;

  /* backprop: http://www.ra.cs.uni-tuebingen.de/SNNS/UserManual/node145.html */

  /* last layer has N feature maps (10 in the case of MNIST), with 1
     value in each feature map
    
     backpropagate the output layer using the output unit formula 
  */
  for(i = 0; i < output_layer->num_feature_maps_; i++)
  {
    double net_j = output_layer->feature_maps_[i].x_[0];
    double t_j = desired_output[i];
    double o_j = output_layer->feature_maps_[i].y_[0];

    double delta_j = DSIGMOID(net_j) * (t_j - o_j);

    for(uint32_t j = 0, k = 0; j < prev_layer->num_feature_maps_; j++)
    {
      /* calculate all delta weights from i to j */
      for(uint32_t k1 = 0; k1 < output_layer->kernel_size_; k1++)
      {
        for(uint32_t k2 = 0; k2 < output_layer->kernel_size_; k2++)
        {
          double o_i = prev_layer->feature_maps_[j].y_[k++];
          output_layer->feature_maps_[i].delta_w_ij_[j](k1, k2) = learning_rate * delta_j * o_i;
        }
      }      
    }
  }

  /* backpropagate the previous layers using the hidden unit formula */
  for(i = layers_.size() - 2; i > 0; i--)
  {
    layers_[i]->backward_propagate(FIRST_ORDER_DV, learning_rate); 
  }
}

void layer::backward_propagate(dv_order_t order, double learning_rate)
{
  for(uint32_t i = 0; i < num_feature_maps_; i++)
  {

  }
}

} // namespace conv_neural_network