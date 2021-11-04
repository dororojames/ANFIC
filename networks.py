import torch
from torch import nn

from context_model import ContextModel
from entropy_models import __CONDITIONS__, EntropyBottleneck
from generalizedivisivenorm import GeneralizedDivisiveNorm
from modules import AugmentedNormalizedFlow, Conv2d, ConvTranspose2d


class CompressesModel(nn.Module):
    """Basic Compress Model"""

    def __init__(self):
        super(CompressesModel, self).__init__()
        self.divisor = None
        self.num_bitstreams = 1

    def named_main_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' not in name:
                yield (name, param)

    def main_parameters(self):
        for _, param in self.named_main_parameters():
            yield param

    def named_aux_parameters(self, prefix=''):
        for name, param in self.named_parameters(prefix=prefix, recurse=True):
            if 'quantiles' in name:
                yield (name, param)

    def aux_parameters(self):
        for _, param in self.named_aux_parameters():
            yield param

    def _cal_base_cdf(self):
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                m._cal_base_cdf()

    def aux_loss(self):
        aux_loss = []
        for m in self.modules():
            if isinstance(m, EntropyBottleneck):
                aux_loss.append(m.aux_loss())

        return torch.stack(aux_loss).sum() if len(aux_loss) else torch.zeros(1, device=next(self.parameters()).device)


class FactorizedCoder(CompressesModel):
    """FactorizedCoder"""

    def __init__(self, num_priors, quant_mode='noise'):
        super(FactorizedCoder, self).__init__()
        self.analysis = nn.Sequential()
        self.synthesis = nn.Sequential()

        self.entropy_bottleneck = EntropyBottleneck(
            num_priors, quant_mode=quant_mode)

        self.divisor = 16


class HyperPriorCoder(FactorizedCoder):
    """HyperPrior Coder"""

    def __init__(self, num_condition, num_priors, use_mean=False, use_abs=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(HyperPriorCoder, self).__init__(
            num_priors, quant_mode=quant_mode)
        self.use_mean = use_mean
        self.use_abs = not self.use_mean or use_abs
        self.conditional_bottleneck = __CONDITIONS__[condition](
            use_mean=use_mean, quant_mode=quant_mode)
        if use_context:
            self.conditional_bottleneck = ContextModel(
                num_condition, num_condition*2, self.conditional_bottleneck)
        self.hyper_analysis = nn.Sequential()
        self.hyper_synthesis = nn.Sequential()

        self.divisor = 64
        self.num_bitstreams = 2


class GoogleHyperAnalysisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperAnalysisTransform, self).__init__(
            Conv2d(num_features, num_filters, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_filters, kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            Conv2d(num_filters, num_hyperpriors, kernel_size=5, stride=2)
        )


class GoogleHyperSynthesisTransform(nn.Sequential):
    def __init__(self, num_features, num_filters, num_hyperpriors):
        super(GoogleHyperSynthesisTransform, self).__init__(
            ConvTranspose2d(num_hyperpriors, num_filters,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters, num_filters * 3 // 2,
                            kernel_size=5, stride=2),
            nn.ReLU(inplace=True),
            ConvTranspose2d(num_filters * 3 // 2, num_features,
                            kernel_size=3, stride=1)
        )


class AugmentedNormalizedAnalysisTransform(AugmentedNormalizedFlow):
    def __init__(self, in_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedAnalysisTransform, self).__init__(
            Conv2d(in_channels, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_filters, kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters),
            Conv2d(num_filters, num_features *
                   (2 if use_affine else 1), kernel_size, stride=2),
            nn.Identity(),
            use_affine=use_affine, transpose=False, distribution=distribution
        )


class AugmentedNormalizedSynthesisTransform(AugmentedNormalizedFlow):
    def __init__(self, out_channels, num_features, num_filters, kernel_size, use_affine, distribution):
        super(AugmentedNormalizedSynthesisTransform, self).__init__(
            nn.Identity(),
            ConvTranspose2d(num_features, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, num_filters,
                            kernel_size, stride=2),
            GeneralizedDivisiveNorm(num_filters, inverse=True),
            ConvTranspose2d(num_filters, out_channels *
                            (2 if use_affine else 1), kernel_size, stride=2),
            use_affine=use_affine, transpose=True, distribution=distribution
        )


class DQ_ResBlock(nn.Sequential):
    def __init__(self, num_filters):
        super().__init__(
            Conv2d(num_filters, num_filters, 3),
            nn.LeakyReLU(0.2, inplace=True),
            Conv2d(num_filters, num_filters, 3)
        )

    def forward(self, input):
        return super().forward(input) + input


class DeQuantizationModule(nn.Module):

    def __init__(self, in_channels, out_channels, num_filters, num_layers):
        super(DeQuantizationModule, self).__init__()
        self.conv1 = Conv2d(in_channels, num_filters, 3)
        self.resblock = nn.Sequential(
            *[DQ_ResBlock(num_filters) for _ in range(num_layers)])
        self.conv2 = Conv2d(num_filters, num_filters, 3)
        self.conv3 = Conv2d(num_filters, out_channels, 3)

    def forward(self, input):
        conv1 = self.conv1(input)
        x = self.resblock(conv1)
        conv2 = self.conv2(x) + conv1
        conv3 = self.conv3(conv2) + input

        return conv3


class AugmentedNormalizedFlowHyperPriorCoder(HyperPriorCoder):
    """AugmentedNormalizedFlowHyperPriorCoder"""

    def __init__(self, num_filters, num_features, num_hyperpriors,
                 in_channels=3, out_channels=3, kernel_size=5, num_layers=1,
                 init_code='gaussian', use_QE=False, use_affine=True,
                 hyper_filters=192, use_mean=False, use_context=False,
                 condition='Gaussian', quant_mode='noise'):
        super(AugmentedNormalizedFlowHyperPriorCoder, self).__init__(
            num_features, num_hyperpriors, use_mean, False, use_context, condition, quant_mode)
        self.num_layers = num_layers
        if not isinstance(num_filters, list):
            num_filters = [num_filters]
        if len(num_filters) != num_layers:
            num_filters = [num_filters[0]] * num_layers

        self.__delattr__('analysis')
        self.__delattr__('synthesis')

        for i in range(num_layers):
            self.add_module('analysis'+str(i), AugmentedNormalizedAnalysisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and init_code != 'zeros', distribution=init_code))
            self.add_module('synthesis'+str(i), AugmentedNormalizedSynthesisTransform(
                in_channels, num_features, num_filters[i], kernel_size, use_affine=use_affine and i != num_layers-1, distribution=init_code))

        self.hyper_analysis = GoogleHyperAnalysisTransform(
            num_features, hyper_filters, num_hyperpriors)

        if use_context:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*2, hyper_filters, num_hyperpriors)
        elif self.use_mean or "Mixture" in condition:
            self.hyper_synthesis = GoogleHyperSynthesisTransform(
                num_features*self.conditional_bottleneck.condition_size, hyper_filters, num_hyperpriors)
        else:
            pass

        if use_QE:
            self.QE = DeQuantizationModule(in_channels, in_channels, 64, 6)
        else:
            self.QE = None

    def __getitem__(self, key):
        return self.__getattr__(key)

    def encode(self, input, code=None, jac=None):
        for i in range(self.num_layers):
            _, code, jac = self['analysis'+str(i)](input, code, jac)

            if i < self.num_layers-1:
                input, _, jac = self['synthesis'+str(i)](input, code, jac)

        return input, code, jac

    def decode(self, input, code=None, jac=None):
        for i in range(self.num_layers-1, -1, -1):
            input, _, jac = self['synthesis'+str(i)](
                input, code, jac, rev=True, last_layer=i == self.num_layers-1)

            if i or jac is not None:
                _, code, jac = self['analysis' +
                                    str(i)](input, code, jac, rev=True)

        return input, code, jac

    def entropy_model(self, input, code, jac=False):
        # Enrtopy coding

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True)

        return Y_error, y_tilde, z_tilde, y_likelihood, z_likelihood

    def compress(self, input, code=None, return_hat=False):
        input, code, _ = self.encode(input, code, jac=None)

        hyperpriors = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        side_stream, h_hat = self.entropy_bottleneck.compress(
            hyperpriors, return_sym=True)

        condition = self.hyper_synthesis(h_hat)

        ret = self.conditional_bottleneck.compress(
            code, condition=condition, return_sym=return_hat)

        if return_hat:
            stream, z_hat = ret

            x_hat = self.decode(None, z_hat, jac=None)[0]

            if self.QE is not None:
                x_hat = self.QE(x_hat)

            return x_hat, [stream, side_stream], [z_hat.size(), h_hat.size()]
        else:
            stream = ret
            return [stream, side_stream], [code.size(), h_hat.size()]

    def decompress(self, strings, shapes):
        stream, side_stream = strings
        z_shape, h_shape = shapes

        h_hat = self.entropy_bottleneck.decompress(side_stream, h_shape)

        condition = self.hyper_synthesis(h_hat)

        z_hat = self.conditional_bottleneck.decompress(
            stream, z_shape, condition=condition)

        reconstructed = self.decode(None, z_hat, jac=None)[0]

        if self.QE is not None:
            reconstructed = self.QE(reconstructed)

        return reconstructed

    def forward(self, input, code=None, jac=None):
        # Encode

        ori_input = input
        jac = [] if jac else None

        input, code, jac = self.encode(input, code, jac)

        # Enrtopy coding

        hyper_code = self.hyper_analysis(
            code.abs() if self.use_abs else code)

        z_tilde, z_likelihood = self.entropy_bottleneck(hyper_code)
        # z_tilde = hyper_code

        condition = self.hyper_synthesis(z_tilde)

        y_tilde, y_likelihood = self.conditional_bottleneck(
            code, condition=condition)
        # y_tilde = code

        # Encode distortion
        Y_error, _, jac = self['synthesis'+str(self.num_layers-1)](
            input, y_tilde, jac, last_layer=True)

        input, code, hyper_code = None, y_tilde, z_tilde
        # input, code, hyper_code = Y_error, y_tilde, z_tilde
        # input = Y_error

        # Decode
        input, code, jac = self.decode(input, code, jac)

        if self.QE is not None:
            input = self.QE(input)

        return input, (y_likelihood, z_likelihood), Y_error
