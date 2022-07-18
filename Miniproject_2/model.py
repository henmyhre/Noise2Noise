import torch
from torch.nn.functional import fold, unfold
import math
import pickle
from pathlib import Path

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# For the Conv2d we inspired from https://coolgpu.github.io/coolgpu_blog/github/pages/2020/10/04/convolution.html
class Conv2d (object):
    """
    Class for 2D-convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1):
        """
        Initializes the 2D-convolution with the provided parameters.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        #Initialize weights and biases
        k = math.sqrt(1/(in_channels*kernel_size*kernel_size))
        self.weight = torch.empty(
            out_channels, in_channels, kernel_size, kernel_size).data.uniform_(-k, k)

        self.bias = torch.empty(
            out_channels).data.uniform_(-k, k)

    def forward(self, input):
        """
        Forward function to reshape the input using the weights and biases.
        """

        self.input = input
        self.num_batches, _, self.h_in, self.w_in = self.input.shape

        # Planning in advance the out-dimension
        self.h_out = (self.h_in+2*self.padding - self.dilation *
                      (self.kernel_size-1)-1)//self.stride + 1
        self.w_out = (self.w_in+2*self.padding - self.dilation *
                      (self.kernel_size-1)-1)//self.stride + 1

        # torch.unfold expands the patches of the input wrt to the Kernel
        # self.input shape : (num_batches, channels_in , w_in , h_im )
        # input_exp's expanded shape : (num_batches , in_channels * k^2 , num_pix = w_out * h_out  ) -> it's huge !
        # num_pix is the number of out pixels

        input_exp = unfold(self.input, kernel_size=self.kernel_size,
                           padding=self.padding, stride=self.stride, dilation=self.dilation)

        # reshape the unfolded data and kernel to so that the last dimension of input_exp_T
        # and the first dimension of weight_T have the same size for matrix multiplication.
        input_exp_T = input_exp.transpose(1, 2)
        weight_T = self.weight.reshape(self.out_channels, -1).t()

        # Tensor multiplication !! Big moment !!
        output_before_reshape = input_exp_T.matmul(weight_T)

        # Transposing and reshaping to output dimensions :
        output_mid_reshape = output_before_reshape.transpose(1, 2)
        output_reshaped = output_mid_reshape.reshape(
            self.num_batches, self.out_channels, self.h_out, self.w_out)

        # Saving information for backpropagation
        self.weight_T = weight_T
        self.input_exp_T = input_exp_T

        output_reshaped += self.bias.view(1, -1, 1, 1)

        return output_reshaped

    def backward(self, gradwrtoutput):
        """
        Computes the gradient wrt weights, biases and input. Saves gradient wrt weights and biases in the object, and returns the gradient wrt input.
        """

        self.grad_bias = gradwrtoutput.sum(dim=[0, 2, 3])  # computes gradient wrt bias

        # upstream gradient has shape : ( batch_num , channels_out , rows , columns)
        # but it is reshaped to have : ( batch_num , num_pix = rows * cols , channels_out )
        new_shape = (self.num_batches, self.out_channels,
                     self.w_out * self.h_out)

        gradient_reshaped = gradwrtoutput.reshape(new_shape).transpose(1, 2)

        # BIG TIME : tensor multiplication !
        grad_weight_1 = gradient_reshaped.matmul(
            self.weight_T.t()).transpose(1, 2)

        # more reshaping and calculation :
        grad_weight_shape = (self.out_channels, self.in_channels,
                             self.kernel_size, self.kernel_size)
        # Compute and save the gradient wrt weights
        self.grad_weight = self.input_exp_T.transpose(1, 2).matmul(
            gradient_reshaped).sum(dim=0).t().view(grad_weight_shape)

        # Folding in back together afterapplyig the tensor multiplication
        grad_input = fold(grad_weight_1, (self.h_in, self.w_in), (self.kernel_size,
                          self.kernel_size), padding=self.padding, stride=self.stride, dilation=self.dilation)

        return grad_input

    def param(self):
        """
        Get weigths and biases.
        """
        return [self.weight, self.bias]

    def to(self, device):
        """
        Transform weight and bias tensors to the provided device.
        """
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)


class ReLU (object):
    """
    Class to perform ReLU operations.
    """
    def forward(self, input):
        """
        Calculates the ReLU on the input.
        """
        self.cache = (input > 0)
        return input * self.cache

    def backward(self, gradwrtoutput):
        """
        Calculate the derivative of ReLU wrt input multiplied by gradwrtoutput.
        """
        return self.cache * gradwrtoutput

    def param(self):
        return []


class Sigmoid (object):
    """
    Class to perform Sigmoid operations.
    """
    def forward(self, input):
        """
        Calculates the Sigmoid on the input.
        """
        self.cache = input.sigmoid()
        return self.cache

    def backward(self, gradwrtoutput):
        """
        Calculate the derivative of Sigmoid wrt input multiplied by gradwrtoutput.
        """
        return self.cache * self.cache.multiply(-1).add(1) * gradwrtoutput

    def param(self):
        return []


class MSE(object):
    """
    Class to compute the MSE-loss.
    """
    def forward(self, y_truth, y_predict):
        """
        Computes the MSE-loss by comparing the actual values with the predicted values.
        """
        self.y_truth_cache = y_truth
        self.y_predict_cache = y_predict
        return (self.y_truth_cache - self.y_predict_cache).square().mean()

    def backward(self):
        """
        Calculate the derivative of MSE wrt y_predict multiplied by gradwrtoutput (while calculating: dividing by the multiplication of all dimensions).
        """
        acc = 1
        for dim in self.y_predict_cache.shape:
            acc *= dim
        return (self.y_predict_cache - self.y_truth_cache).mul(2).div(acc)

    def param(self):
        return []


class SGD (object):
    """
    Class to perform optimization with Stochastic Gradient Descent. 
    """
    def __init__(self, layers, lr):
        """
        Initializes with provided learning rate and the layers from the models network.
        """
        self.lr = lr
        self.layers = layers

    def update(self):
        """
        Updates the weights and biases for each layer.
        """
        for layer in self.layers:
            # update layer
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                layer.weight -= self.lr * layer.grad_weight
                layer.bias -= self.lr * layer.grad_bias


class TransposeConv2d(object):
    """
    Class for transpose convolution.
    """
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride=1, padding=0, dilation=1):
        """
        Initialize the transpose convolution.
        """

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.dilation = dilation

        # Initialize the weigths and biases
        k = math.sqrt(1/(out_channels*kernel_size*kernel_size))
        self.weight = torch.empty(
            in_channels, out_channels, kernel_size, kernel_size).data.uniform_(-k, k)

        self.bias = torch.empty(
            out_channels).data.uniform_(-k, k)

    def forward(self, input):
        """
        Forward function to reshape the input using the weights and biases.
        """
        self.input = input
        num_batches, _, h_in, w_in = self.input.shape

        # Calculate new image dimensions
        h_out = (h_in - 1) * self.stride - 2 * self.padding + \
            self.dilation * (self.kernel_size - 1) + 1
        w_out = (w_in - 1) * self.stride - 2 * self.padding + \
            self.dilation * (self.kernel_size - 1) + 1

        new_input_shape = (num_batches, self.in_channels, h_in * w_in)

        input_T = self.input.reshape(new_input_shape).transpose(1, 2)
        weight_T = self.weight.reshape(self.in_channels, -1)

        # Multiply input with weights
        weight_input = input_T.matmul(weight_T).transpose(1, 2)

        # Reshape to output dimensions
        out = fold(weight_input, (h_out, w_out), kernel_size=self.kernel_size,
                   padding=self.padding, stride=self.stride, dilation=self.dilation)
        self.input_T = input_T

        # Add bias
        out += self.bias.view(1, -1, 1, 1)

        return out

    def backward(self, gradwrtoutput):
        """
        Computes the gradient wrt weights, biases and input. Saves gradient wrt weights and biases in the object, and returns the gradient wrt input.
        """
        num_batches, _, h_in, w_in = self.input.shape
        grad_output = unfold(gradwrtoutput, kernel_size=self.kernel_size,
                             padding=self.padding, stride=self.stride, dilation=self.dilation)

        # Get and save the gradient wrt bias
        self.grad_bias = gradwrtoutput.sum(dim=[0, 2, 3])

        # Reshaping
        grad_output_T = grad_output.transpose(1, 2)
        weight_T = self.weight.reshape(self.in_channels, -1).t()

        # Calculate and save the gradient wrt weigths
        self.grad_weight = self.input_T.transpose(1, 2).matmul(
            grad_output_T).sum(dim=0).reshape(self.weight.shape)

        # Multiply the output with weights
        grad_input_before_reshape = grad_output_T.matmul(
            weight_T).transpose(1, 2)

        # Reshape to correct dimensions
        grad_input = grad_input_before_reshape.reshape(
            num_batches, self.in_channels, h_in, w_in)
        return grad_input

    def param(self):
        """
        Get the weights and biases.
        """
        return [self.weight, self.bias]

    def to(self, device):
        """
        Transform weight and bias tensors to the provided device.
        """
        self.weight = self.weight.to(device)
        self.bias = self.bias.to(device)


Upsampling = TransposeConv2d # Upsampling using transpse convolution.


class Sequential(object):
    """
    Creates the network and performs forward and backward pass on this.
    """
    def __init__(self, *layers):
        """
        Initialize the network with the provided layers.
        """
        self.layers = layers

    def set_loss(self, loss_type):
        """
        Set the loss class.
        """
        self.loss_class = loss_type

    def set_optimizer(self, optimizer):
        """
        Set the optimizer.
        """
        self.optimizer = optimizer

    def forward(self, input_data):
        """
        Forward pass on all the layers and returning th final output.
        """
        output = input_data
        for layer in self.layers:
            output = layer.forward(output)
        return output

    def backward(self, grad_wrt_output):
        """
        Backward pass on all the layers to find the gradients.
        """
        for layer in reversed(self.layers):
            grad_wrt_output = layer.backward(grad_wrt_output)

    def to(self, device):
        """
        Transform weight and bias tensors to the provided device for all layers.
        """
        for layer in self.layers:
            if hasattr(layer, 'to'):
                layer.to(device)


class Model():
    """
    Class for initializing, training and testing a model.
    """
    def __init__(self) -> None:
        # Defining the model
        self.model = Sequential(
            Conv2d(3, 256, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Conv2d(256, 256, kernel_size=3, stride=2, padding=1),
            ReLU(),
            Upsampling(256, 256, kernel_size=2, stride=2, padding=0),
            ReLU(),
            Upsampling(256, 3, kernel_size=2, stride=2, padding=0),
            Sigmoid()
        )
        # Defining the los function
        self.model.set_loss(MSE())
        # Defining the optimizer
        self.model.set_optimizer(SGD(self.model.layers, 0.8))

        self.model.to(device)

    def load_pretrained_model(self) -> None:
        """
        Loads a saved model from bestmodel.pth.
        """
        model_path = Path(__file__).parent / "bestmodel.pth"
        with open(model_path, "rb") as input_file:
            loadedLayers = iter(pickle.load(input_file))
            for layer in self.model.layers:
                if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                    loadedParams = next(loadedLayers)
                    layer.weight = loadedParams['weight'].to(device)
                    layer.bias = loadedParams['bias'].to(device)

    def save_model(self) -> None:
        """
        Saves a model to bestmodel.pth.
        """
        layersToBeSaved = []
        for layer in self.model.layers:
            if hasattr(layer, 'weight') and hasattr(layer, 'bias'):
                layersToBeSaved.append(
                    {'weight': layer.weight, 'bias': layer.bias})

        with open("bestmodel.pth", "wb") as output_file:
            pickle.dump(layersToBeSaved, output_file)

    def train(self, train_input, train_target, num_epochs) -> None:
        """
        Trains model on the provided training input and prints its loss for each epoch.
        """
        batch_size = 8
        num_batches = math.ceil(len(train_input)/batch_size)
        for e in range(num_epochs):
            train_losses = 0
            for i in range(0, len(train_input), batch_size):
                # Get output after forward
                output = self.model.forward(
                    train_input[i:i + batch_size].float().to(device)/255)

                # Compute the loss and the output gradient
                loss = self.model.loss_class.forward(
                    train_target[i:i + batch_size].float().to(device)/255, output)
                output_gradient = self.model.loss_class.backward()

                # Calculate and update the weights and biases
                self.model.backward(output_gradient)
                self.model.optimizer.update()

                # Save the loss for printing
                train_losses += loss.item()
            print('Epoch : ', e+1, '\t', 'train-loss :',
                  train_losses/num_batches)

    def predict(self, test_input) -> torch.Tensor:
        """
        Predicts output on the given test_input.
        """
        input_type = test_input.type()
        denoised = self.model.forward(test_input.float().to(device)/255)*255
        return denoised.clamp(min=0, max=255).type(input_type) # Make sure that the output is between 0-255
