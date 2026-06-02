import torch
import math
import copy
from torch import nn
from typing import Optional
from torch import Tensor


class LaplaceRandomFeatureCovariance(nn.Module):
    """Computes the Gaussian Process covariance using Laplace method.

    At training time, this layer updates the Gaussian process posterior using
    model features in minibatches.

    Attributes:
      momentum: (float) A discount factor used to compute the moving average for
        posterior precision matrix. Analogous to the momentum factor in batch
        normalization. If -1 then update covariance matrix using a naive sum
        without momentum, which is desirable if the goal is to compute the exact
        covariance matrix by passing through data once (say in the final epoch).
      ridge_penalty: (float) Initial Ridge penalty to weight covariance matrix.
        This value is used to stablize the eigenvalues of weight covariance
        estimate so that the matrix inverse can be computed for Cov = inv(t(X) * X
        + s * I). The ridge factor s cannot be too large since otherwise it will
        dominate the t(X) * X term and make covariance estimate not meaningful.
      likelihood: (str) The likelihood to use for computing Laplace approximation
        for the covariance matrix. Can be one of ('binary_logistic', 'poisson',
        'gaussian').
    """

    def __init__(
            self,
            momentum=0.999,
            ridge_penalty=1e-6,
            likelihood='gaussian',
            use_on_read_synchronization_for_single_replica_vars=False,
            gp_feature_dim=1024,
    ):
        super(LaplaceRandomFeatureCovariance, self).__init__()
        _SUPPORTED_LIKELIHOOD = ('binary_logistic', 'poisson', 'gaussian')
        if likelihood not in _SUPPORTED_LIKELIHOOD:
            raise ValueError(
                f'"likelihood" must be one of {_SUPPORTED_LIKELIHOOD}, got {likelihood}.'
            )
        self.ridge_penalty = ridge_penalty
        self.momentum = momentum
        self.likelihood = likelihood
        self.use_on_read_synchronization_for_single_replica_vars = use_on_read_synchronization_for_single_replica_vars
        # Track this tensor, move it to the GPU with .to(device), include it in the state_dict, but do not treat it as a parameter for optimization or graph analysis."
        self.register_buffer('initial_precision_matrix', torch.zeros(gp_feature_dim, gp_feature_dim))
        self.register_buffer('precision_matrix', torch.zeros(gp_feature_dim, gp_feature_dim))
        self.register_buffer('covariance_matrix', torch.zeros(gp_feature_dim, gp_feature_dim))
        self.gp_feature_dim = gp_feature_dim


    def update_feature_precision_matrix(self, gp_feature, logits):
        """Computes the updated precision matrix of feature weights."""
        gp_feature_detached = gp_feature.detach()

        if self.likelihood != 'gaussian':
            if logits is None:
                raise ValueError(
                    f'"logits" cannot be None when likelihood={self.likelihood}')

            if logits.shape[-1] != 1:
                raise ValueError(
                    f'likelihood={self.likelihood} only support univariate logits.'
                    f'Got logits dimension: {logits.shape[-1]}')

        batch_size = gp_feature_detached.shape[0]
        # Computes batch-specific normalized precision matrix.
        if self.likelihood == 'binary_logistic':
            prob = torch.sigmoid(logits)
            prob_multiplier = prob * (1. - prob)
        elif self.likelihood == 'poisson':
            prob_multiplier = torch.exp(logits)
        else:
            # prob_multiplier = torch.ones(1, device=gp_feature.device)
            prob_multiplier = 1
            pass
        gp_feature_adjusted = math.sqrt(prob_multiplier) * gp_feature_detached
        precision_matrix_minibatch = torch.matmul(gp_feature_adjusted.t(), gp_feature_adjusted)
        # Updates the population-wise precision matrix.
        if self.momentum > 0:
            # Use moving-average updates to accumulate batch-specific precision matrices.
            precision_matrix_minibatch = precision_matrix_minibatch / batch_size
            precision_matrix_new = (
                    self.momentum * self.precision_matrix +
                    (1. - self.momentum) * precision_matrix_minibatch)
            self.precision_matrix.copy_(precision_matrix_new)
        else:
            # Compute exact population-wise covariance without momentum.
            # If use this option, make sure to pass through data only once.
            self.precision_matrix.add_(precision_matrix_minibatch)

    @torch._dynamo.disable
    def reset_precision_matrix(self):
        """Resets precision matrix to its initial value.

        This function is useful for reseting the model's covariance matrix at the
        begining of a new epoch.
        """
        self.precision_matrix.copy_(self.initial_precision_matrix.to(self.precision_matrix.device))

    @torch._dynamo.disable
    def update_feature_covariance_matrix(self):
        """Computes the feature covariance"""
        new_cov = torch.linalg.inv(self.precision_matrix + self.ridge_penalty *
                                   torch.eye(self.gp_feature_dim, device=self.precision_matrix.device))
        self.covariance_matrix.copy_(new_cov)

    def compute_predictive_covariance(self, gp_feature):
        """Computes posterior predictive variance.

        Approximates the Gaussian process posterior using random features.
        Given training random feature Phi_tr (num_train, num_hidden) and testing
        random feature Phi_ts (batch_size, num_hidden). The predictive covariance
        matrix is computed as (assuming Gaussian likelihood):

        s * Phi_ts @ inv(t(Phi_tr) * Phi_tr + s * I) @ t(Phi_ts),

        where s is the ridge factor to be used for stablizing the inverse, and I is
        the identity matrix with shape (num_hidden, num_hidden).

        Args:
          gp_feature: (torch.Tensor) The random feature of testing data to be used for
            computing the covariance matrix. Shape (batch_size, gp_hidden_size).

        Returns:
          (torch.Tensor) Predictive covariance matrix, shape (batch_size, batch_size).
        """

        # Computes the covariance matrix of the gp prediction.
        cov_feature_product = torch.matmul(self.covariance_matrix, gp_feature.t()) * self.ridge_penalty
        gp_cov_matrix = torch.matmul(gp_feature, cov_feature_product)
        return gp_cov_matrix

    def forward(self, gp_feature, logits: Optional[Tensor] = None,
                update_precision_matrix: bool = True,
                update_covariance_matrix: bool = True,
                return_covariance: bool = True):
        if update_precision_matrix:
            self.update_feature_precision_matrix(gp_feature, logits)
        if update_covariance_matrix:
            self.update_feature_covariance_matrix()
        if return_covariance:
            return self.compute_predictive_covariance(gp_feature)
        else:
            return torch.eye(gp_feature.shape[0], device=gp_feature.device)
class CosModule(nn.Module):
    def forward(self, x):
        return torch.cos(x)

class RandomFeatureGaussianProcess(nn.Module):
    """Gaussian process layer with random feature approximation.

    During training, the model updates the maximum a posteriori (MAP) logits
    estimates and posterior precision matrix using minibatch statistics. During
    inference, the model divides the MAP logit estimates by the predictive
    standard deviation, which is equivalent to approximating the posterior mean
    of the predictive probability via the mean-field approximation.

    User can specify different types of random features by setting
    `use_custom_random_features=True`, and change the initializer and activations
    of the custom random features. For example:

      MLP Kernel: initializer='random_normal', activation=tf.nn.relu
      RBF Kernel: initializer='random_normal', activation=tf.math.cos

    A linear kernel can also be specified by setting gp_kernel_type='linear' and
    `use_custom_random_features=True`.

    Attributes:
      units: (int) The dimensionality of layer.
      num_inducing: (int) The number of random features for the approximation.
      is_training: (tf.bool) Whether the layer is set in training mode. If so the
        layer updates the Gaussian process' variance estimate using statistics
        computed from the incoming minibatches.
    """

    def __init__(self,
                 num_class,
                 gp_hidden_dim=1024,
                 num_inducing=1024,
                 gp_kernel_type='gaussian',
                 gp_kernel_scale=1.,
                 gp_output_bias=0.,
                 normalize_input=False,
                 gp_kernel_scale_trainable=False,
                 gp_output_bias_trainable=False,
                 gp_cov_momentum=0.999,
                 gp_cov_ridge_penalty=1e-6,
                 scale_random_features=False,
                 use_custom_random_features=True,
                 custom_random_features_initializer="orf",
                 gp_cov_likelihood='gaussian',
                 gp_output_imagenet_initializer=False,):
        """Initializes a random-feature Gaussian process layer instance.

        Args:
          units: (int) Number of output units.
          gp_hidden_dim: (int) Number of input features for Gaussian process.
          num_inducing: (int) Number of random Fourier features used for
            approximating the Gaussian process.
          gp_kernel_type: (string) The type of kernel function to use for Gaussian
            process. Currently default to 'gaussian' which is the Gaussian RBF
            kernel.
          gp_kernel_scale: (float) The length-scale parameter of the a
            shift-invariant kernel function, i.e., for RBF kernel:
            exp(-|x1 - x2|**2 / gp_kernel_scale).
          gp_output_bias: (float) Scalar initial value for the bias vector.
          normalize_input: (bool) Whether to normalize the input to Gaussian
            process.
          gp_kernel_scale_trainable: (bool) Whether the length scale variable is
            trainable.
          gp_output_bias_trainable: (bool) Whether the bias is trainable.
          gp_cov_momentum: (float) The discount factor to compute the moving average of precision matrix.'
             If -1 then instead compute the exact covariance at the lastest epoch. if set to -1, should reset
             covariance matrix at the beginning of each epoch.
          gp_cov_ridge_penalty: (float) Initial Ridge penalty to posterior
            covariance matrix.
          scale_random_features: (bool) Whether to scale the random feature
            by sqrt(2. / num_inducing).
          use_custom_random_features: (bool) Whether to use custom random
            features implemented using tf.keras.layers.Dense.
          custom_random_features_initializer: (str) Initializer for
            the random features. Default to random normal which approximates a RBF
            kernel function if activation function is cos.
          gp_cov_likelihood: (string) Likelihood to use for computing Laplace
            approximation for covariance matrix. Default to `gaussian`.
        """
        super(RandomFeatureGaussianProcess, self).__init__()
        self.gp_hidden_dim = gp_hidden_dim
        self.num_inducing = num_inducing
        if gp_kernel_type == 'linear':
            self.num_inducing = self.gp_hidden_dim

        self.normalize_input = normalize_input
        self.gp_input_scale = (
            1. / math.sqrt(gp_kernel_scale) if gp_kernel_scale is not None else None)
        self.gp_feature_scale = math.sqrt(2. / float(num_inducing))

        self.scale_random_features = scale_random_features

        self.gp_kernel_type = gp_kernel_type
        self.gp_kernel_scale = gp_kernel_scale
        self.gp_output_bias = gp_output_bias
        self.gp_kernel_scale_trainable = gp_kernel_scale_trainable
        self.gp_output_bias_trainable = gp_output_bias_trainable

        self.use_custom_random_features = use_custom_random_features
        self.custom_random_features_initializer = custom_random_features_initializer

        self.gp_cov_momentum = gp_cov_momentum
        self.gp_cov_ridge_penalty = gp_cov_ridge_penalty
        self.gp_cov_likelihood = gp_cov_likelihood

        self.covariance_layer = LaplaceRandomFeatureCovariance(
            momentum=gp_cov_momentum,
            ridge_penalty=gp_cov_ridge_penalty,
            likelihood=gp_cov_likelihood,
            gp_feature_dim=self.num_inducing,
        )
        # if normalize_input:
        self._input_norm_layer = nn.LayerNorm(self.gp_hidden_dim)
        self._random_feature = self._make_random_feature_layer()

        self._gp_output_layer = nn.Linear(self.num_inducing, num_class, bias=False)
        self._gp_output_bias = nn.Parameter(torch.zeros(num_class), requires_grad=self.gp_output_bias_trainable)
        if gp_output_imagenet_initializer:
            nn.init.normal_(self._gp_output_layer.weight, mean=0.0, std=0.01)

    def reset_covariance_matrix(self):
        self.covariance_layer.reset_precision_matrix()

    def update_feature_precision_matrix(self, gp_feature, gp_output):
        self.covariance_layer.update_feature_precision_matrix(gp_feature, gp_output)

    def update_covariance_matrix(self):
        self.covariance_layer.update_feature_covariance_matrix()

    def compute_predictive_covariance(self, gp_feature):
        return self.covariance_layer.compute_predictive_covariance(gp_feature)

    def _make_random_feature_layer(self):
        """Defines random feature layer depending on kernel type."""
        if not self.use_custom_random_features:
            # Use default RandomFourierFeatures layer from tf.keras.
            raise NotImplementedError("Default tensorflow implement is tf.keras.layers.experimental.RandomFourierFeatures, "
                                      "but it is not implemented in PyTorch. Please use custom random features.")
        if self.gp_kernel_type.lower() == 'linear':
            custom_random_feature_layer = nn.Identity()
        else:
            # Use user-supplied configurations.
            custom_random_feature_layer = nn.Sequential(nn.Linear(self.gp_hidden_dim, self.num_inducing, bias=True),
                                                         CosModule())
            custom_random_feature_layer.requires_grad_(False)
            nn.init.uniform_(custom_random_feature_layer[0].bias, a=0.0, b=2. * math.pi)
            if self.custom_random_features_initializer == "orf":
                nn.init.orthogonal_(custom_random_feature_layer[0].weight, gain=0.05)
            elif self.custom_random_features_initializer == "rff":
                nn.init.normal_(custom_random_feature_layer[0].weight, mean=0.0, std=0.05)
            else:
                raise ValueError(f"Unknown custom random feature initializer: {self.custom_random_features_initializer}")
        return custom_random_feature_layer

    def forward(self, inputs,
                update_precision_matrix: bool = True,
                update_covariance_matrix: bool = False,
                return_random_features: bool = False,
                return_covariance: bool = False):
        # rescaling the input.
        # Computes random features.
        gp_inputs = inputs
        if self.normalize_input:
            gp_inputs = self._input_norm_layer(gp_inputs)
        elif self.use_custom_random_features and self.gp_input_scale is not None:
            # Supports lengthscale for custom random feature layer by directly rescaling the input.
            gp_inputs = gp_inputs * self.gp_input_scale

        gp_feature = self._random_feature(gp_inputs)

        if self.scale_random_features:
            # Scale random feature by 2. / sqrt(num_inducing) following [1].
            # When using GP layer as the output layer of a neural network,
            # it is recommended to turn this scaling off to prevent it from changing
            # the learning rate to the hidden layers.
            gp_feature = gp_feature * self.gp_feature_scale

        # Computes posterior center (i.e., MAP estimate) and variance.
        gp_output = self._gp_output_layer(gp_feature) + self._gp_output_bias
        if update_precision_matrix:
            self.update_feature_precision_matrix(gp_feature, gp_output)

        if update_covariance_matrix:
            self.update_feature_covariance_matrix()
        output = (gp_output, )
        if return_random_features:
            output += (gp_feature,)
        if return_covariance:
            output += (self.compute_predictive_covariance(gp_feature),)
        return output if len(output) !=1 else output[0]


if __name__ == "__main__":
    from torchvision.models import resnet18

    class ResNet18GP(nn.Module):
        def __init__(self):
            super(ResNet18GP, self).__init__()
            self.resnet18 = resnet18(weights="ResNet18_Weights.DEFAULT")
            self.resnet18.fc = nn.Identity()
            self.random_feature_gaussian_process = RandomFeatureGaussianProcess(gp_hidden_dim=512, units=512, num_inducing=512,num_class=3)
        def forward(self, x, **kwargs):
            feature = self.resnet18(x)
            gp_output = self.random_feature_gaussian_process(feature, **kwargs)
            return gp_output
    model = ResNet18GP()
    output = model(torch.randn(10, 3, 224, 224), return_random_features=False, return_covariance=False,
                   update_precision_matrix=False, update_covariance_matrix=False)
    print("length of output:", output.shape)

    # print(output[0].shape, output[1].shape, output[2].shaoe)