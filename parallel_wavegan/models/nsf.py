"""Neural-source filter (NSF) models

Code is adopted from https://github.com/nii-yamagishilab/project-NN-Pytorch-scripts
to match the style with the ParallelWaveGAN respository.
"""

import sys

import numpy as np
import torch
import torch.nn as torch_nn
import torch.nn.functional as torch_nn_func

from parallel_wavegan.layers import upsample


class BLSTMLayer(torch_nn.Module):
    """Wrapper over BLSTM
    Assume hidden layer = 1
    """

    def __init__(self, input_dim, output_dim):
        super(BLSTMLayer, self).__init__()
        if output_dim % 2 != 0:
            print("Output_dim of BLSTMLayer is {:d}".format(output_dim))
            print("BLSTMLayer expects a layer size of even number")
            sys.exit(1)
        # bi-directional LSTM
        self.l_blstm = torch_nn.LSTM(
            input_dim, output_dim // 2, bidirectional=True, batch_first=True
        )

    def forward(self, x):
        """output = fowrard(x)
        input
        -----
          x: tensor  (batchsize=1, length, dim_in)

        output
        ------
          Output: tensor, (batchsize=1, length, dim_out)
        """
        blstm_data, _ = self.l_blstm(x)
        return blstm_data


class Conv1dKeepLength(torch_nn.Conv1d):
    """Wrapper for causal convolution
    Input tensor:  (batchsize=1, length, dim_in)
    Output tensor: (batchsize=1, length, dim_out)
    https://github.com/pytorch/pytorch/issues/1333
    Note: Tanh is optional
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        dilation_s,
        kernel_s,
        causal=False,
        stride=1,
        groups=1,
        bias=True,
        tanh=True,
        pad_mode="constant",
    ):
        super(Conv1dKeepLength, self).__init__(
            input_dim,
            output_dim,
            kernel_s,
            stride=stride,
            padding=0,
            dilation=dilation_s,
            groups=groups,
            bias=bias,
        )

        self.pad_mode = pad_mode

        self.causal = causal
        # input & output length will be the same
        if self.causal:
            # left pad to make the convolution causal
            self.pad_le = dilation_s * (kernel_s - 1)
            self.pad_ri = 0
        else:
            # pad on both sizes
            self.pad_le = dilation_s * (kernel_s - 1) // 2
            self.pad_ri = dilation_s * (kernel_s - 1) - self.pad_le

        if tanh:
            self.l_ac = torch_nn.Tanh()
        else:
            self.l_ac = torch_nn.Identity()

    def forward(self, data):
        # permute to (batchsize=1, dim, length)
        # add one dimension (batchsize=1, dim, ADDED_DIM, length)
        # pad to ADDED_DIM
        # squeeze and return to (batchsize=1, dim, length)
        # https://github.com/pytorch/pytorch/issues/1333
        x = torch_nn_func.pad(
            data.permute(0, 2, 1).unsqueeze(2),
            (self.pad_le, self.pad_ri, 0, 0),
            mode=self.pad_mode,
        ).squeeze(2)
        # tanh(conv1())
        # permmute back to (batchsize=1, length, dim)
        output = self.l_ac(super(Conv1dKeepLength, self).forward(x))
        return output.permute(0, 2, 1)


class MovingAverage(Conv1dKeepLength):
    """Wrapper to define a moving average smoothing layer
    Note: MovingAverage can be implemented using TimeInvFIRFilter too.
          Here we define another Module dicrectly on Conv1DKeepLength
    """

    def __init__(self, feature_dim, window_len, causal=False, pad_mode="replicate"):
        super(MovingAverage, self).__init__(
            feature_dim,
            feature_dim,
            1,
            window_len,
            causal,
            groups=feature_dim,
            bias=False,
            tanh=False,
            pad_mode=pad_mode,
        )
        # set the weighting coefficients
        torch_nn.init.constant_(self.weight, 1 / window_len)
        # turn off grad for this layer
        for p in self.parameters():
            p.requires_grad = False

    def forward(self, data):
        return super(MovingAverage, self).forward(data)


class TimeInvFIRFilter(Conv1dKeepLength):
    """Wrapper to define a FIR filter over Conv1d
    Note: FIR Filtering is conducted on each dimension (channel)
    independently: groups=channel_num in conv1d
    """

    def __init__(self, feature_dim, filter_coef, causal=True, flag_train=False):
        """__init__(self, feature_dim, filter_coef,
                 causal=True, flag_train=False)
        feature_dim: dimension of input data
        filter_coef: 1-D tensor of filter coefficients
        causal: FIR is causal or not (default: true)
        flag_train: whether train the filter coefficients (default false)

        Input data: (batchsize=1, length, feature_dim)
        Output data: (batchsize=1, length, feature_dim)
        """
        super(TimeInvFIRFilter, self).__init__(
            feature_dim,
            feature_dim,
            1,
            filter_coef.shape[0],
            causal,
            groups=feature_dim,
            bias=False,
            tanh=False,
        )

        if filter_coef.ndim == 1:
            # initialize weight using provided filter_coef
            with torch.no_grad():
                tmp_coef = torch.zeros([feature_dim, 1, filter_coef.shape[0]])
                tmp_coef[:, 0, :] = filter_coef
                tmp_coef = torch.flip(tmp_coef, dims=[2])
                self.weight = torch.nn.Parameter(tmp_coef, requires_grad=flag_train)
        else:
            print("TimeInvFIRFilter expects filter_coef to be 1-D tensor")
            print("Please implement the code in __init__ if necessary")
            sys.exit(1)

    def forward(self, data):
        return super(TimeInvFIRFilter, self).forward(data)


class TimeVarFIRFilter(torch_nn.Module):
    """TimeVarFIRFilter
    Given sequences of filter coefficients and a signal, do filtering

    Filter coefs: (batchsize=1, signal_length, filter_order = K)
    Signal:       (batchsize=1, signal_length, 1)

    For batch 0:
     For n in [1, sequence_length):
       output(0, n, 1) = \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)

    Note: filter coef (0, n, :) is only used to compute the output
          at (0, n, 1)
    """

    def __init__(self):
        super(TimeVarFIRFilter, self).__init__()

    def forward(self, signal, f_coef):
        """
        Filter coefs: (batchsize=1, signal_length, filter_order = K)
        Signal:       (batchsize=1, signal_length, 1)

        Output:       (batchsize=1, signal_length, 1)

        For n in [1, sequence_length):
          output(0, n, 1)= \sum_{k=1}^{K} signal(0, n-k, 1)*coef(0, n, k)

        This method may be not efficient:

        Suppose signal [x_1, ..., x_N], filter [a_1, ..., a_K]
        output         [y_1, y_2, y_3, ..., y_N, *, * ... *]
               = a_1 * [x_1, x_2, x_3, ..., x_N,   0, ...,   0]
               + a_2 * [  0, x_1, x_2, x_3, ..., x_N,   0, ...,  0]
               + a_3 * [  0,   0, x_1, x_2, x_3, ..., x_N, 0, ...,  0]
        """
        signal_l = signal.shape[1]
        order_k = f_coef.shape[-1]

        # pad to (batchsize=1, signal_length + filter_order-1, dim)
        padded_signal = torch_nn_func.pad(signal, (0, 0, 0, order_k - 1))

        y = torch.zeros_like(signal)
        # roll and weighted sum, only take [0:signal_length]
        for k in range(order_k):
            y += (
                torch.roll(padded_signal, k, dims=1)[:, 0:signal_l, :]
                * f_coef[:, :, k : k + 1]
            )
        # done
        return y


class SincFilter(torch_nn.Module):
    """SincFilter
    Given the cut-off-frequency, produce the low-pass and high-pass
    windowed-sinc-filters.
    If input cut-off-frequency is (batchsize=1, signal_length, 1),
    output filter coef is (batchsize=1, signal_length, filter_order).
    For each time step in [1, signal_length), we calculate one
    filter for low-pass sinc filter and another for high-pass filter.

    Example:
    import scipy
    import scipy.signal
    import numpy as np

    filter_order = 31
    cut_f = 0.2
    sinc_layer = SincFilter(filter_order)
    lp_coef, hp_coef = sinc_layer(torch.ones(1, 10, 1) * cut_f)

    w, h1 = scipy.signal.freqz(lp_coef[0, 0, :].numpy(), [1])
    w, h2 = scipy.signal.freqz(hp_coef[0, 0, :].numpy(), [1])
    plt.plot(w, 20*np.log10(np.abs(h1)))
    plt.plot(w, 20*np.log10(np.abs(h2)))
    plt.plot([cut_f * np.pi, cut_f * np.pi], [-100, 0])
    """

    def __init__(self, filter_order):
        super(SincFilter, self).__init__()
        # Make the filter oder an odd number
        #  [-(M-1)/2, ... 0, (M-1)/2]
        #
        self.half_k = (filter_order - 1) // 2
        self.order = self.half_k * 2 + 1

    def hamming_w(self, n_index):
        """prepare hamming window for each time step
        n_index (batchsize=1, signal_length, filter_order)
            For each time step, n_index will be [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 0, :] = [-(M-1)/2, ... 0, (M-1)/2]
            n_index[0, 1, :] = [-(M-1)/2, ... 0, (M-1)/2]
            ...
        output  (batchsize=1, signal_length, filter_order)
            output[0, 0, :] = hamming_window
            output[0, 1, :] = hamming_window
            ...
        """
        # Hamming window
        return 0.54 + 0.46 * torch.cos(2 * np.pi * n_index / self.order)

    def sinc(self, x):
        """Normalized sinc-filter sin( pi * x) / pi * x
        https://en.wikipedia.org/wiki/Sinc_function

        Assume x (batchsize, signal_length, filter_order) and
        x[0, 0, :] = [-half_order, - half_order+1, ... 0, ..., half_order]
        x[:, :, self.half_order] -> time index = 0, sinc(0)=1
        """
        y = torch.zeros_like(x)
        y[:, :, 0 : self.half_k] = torch.sin(np.pi * x[:, :, 0 : self.half_k]) / (
            np.pi * x[:, :, 0 : self.half_k]
        )
        y[:, :, self.half_k + 1 :] = torch.sin(np.pi * x[:, :, self.half_k + 1 :]) / (
            np.pi * x[:, :, self.half_k + 1 :]
        )
        y[:, :, self.half_k] = 1
        return y

    def forward(self, cut_f):
        """lp_coef, hp_coef = forward(self, cut_f)
        cut-off frequency cut_f (batchsize=1, length, dim = 1)

        lp_coef: low-pass filter coefs  (batchsize, length, filter_order)
        hp_coef: high-pass filter coefs (batchsize, length, filter_order)
        """
        # create the filter order index
        with torch.no_grad():
            # [- (M-1) / 2, ..., 0, ..., (M-1)/2]
            lp_coef = torch.arange(-self.half_k, self.half_k + 1, device=cut_f.device)
            # [[[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ],
            #  [[- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   [- (M-1) / 2, ..., 0, ..., (M-1)/2],
            #   ...
            #  ]]
            lp_coef = lp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)

            hp_coef = torch.arange(-self.half_k, self.half_k + 1, device=cut_f.device)
            hp_coef = hp_coef.repeat(cut_f.shape[0], cut_f.shape[1], 1)

            # temporary buffer of [-1^n] for gain norm in hp_coef
            tmp_one = torch.pow(-1, hp_coef)

        # unnormalized filter coefs with hamming window
        lp_coef = cut_f * self.sinc(cut_f * lp_coef) * self.hamming_w(lp_coef)

        hp_coef = (
            self.sinc(hp_coef) - cut_f * self.sinc(cut_f * hp_coef)
        ) * self.hamming_w(hp_coef)

        # normalize the coef to make gain at 0/pi is 0 dB
        # sum_n lp_coef[n]
        lp_coef_norm = torch.sum(lp_coef, axis=2).unsqueeze(-1)
        # sum_n hp_coef[n] * -1^n
        hp_coef_norm = torch.sum(hp_coef * tmp_one, axis=2).unsqueeze(-1)

        lp_coef = lp_coef / lp_coef_norm
        hp_coef = hp_coef / hp_coef_norm

        # return normed coef
        return lp_coef, hp_coef


class UpSampleLayer(torch_nn.Module):
    """Wrapper over up-sampling
    Input tensor: (batchsize=1, length, dim)
    Ouput tensor: (batchsize=1, length * up-sampling_factor, dim)
    """

    def __init__(self, feature_dim, up_sampling_factor, smoothing=False):
        super(UpSampleLayer, self).__init__()
        # wrap a up_sampling layer
        self.scale_factor = up_sampling_factor
        self.l_upsamp = torch_nn.Upsample(scale_factor=self.scale_factor)
        if smoothing:
            self.l_ave1 = MovingAverage(feature_dim, self.scale_factor)
            self.l_ave2 = MovingAverage(feature_dim, self.scale_factor)
        else:
            self.l_ave1 = torch_nn.Identity()
            self.l_ave2 = torch_nn.Identity()
        return

    def forward(self, x):
        # permute to (batchsize=1, dim, length)
        up_sampled_data = self.l_upsamp(x.permute(0, 2, 1))

        # permute it backt to (batchsize=1, length, dim)
        # and do two moving average
        return self.l_ave1(self.l_ave2(up_sampled_data.permute(0, 2, 1)))


# Neural filter block (1 block)
class NeuralFilterBlock(torch_nn.Module):
    """Wrapper over a single filter block"""

    def __init__(self, signal_size, hidden_size, kernel_size=3, conv_num=10):
        super(NeuralFilterBlock, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv_num = conv_num
        self.dilation_s = [np.power(2, x) for x in np.arange(conv_num)]

        # ff layer to expand dimension
        self.l_ff_1 = torch_nn.Linear(signal_size, hidden_size, bias=False)
        self.l_ff_1_tanh = torch_nn.Tanh()

        # dilated conv layers
        tmp = [
            Conv1dKeepLength(
                hidden_size, hidden_size, x, kernel_size, causal=True, bias=False
            )
            for x in self.dilation_s
        ]
        self.l_convs = torch_nn.ModuleList(tmp)

        # ff layer to de-expand dimension
        self.l_ff_2 = torch_nn.Linear(hidden_size, hidden_size // 4, bias=False)
        self.l_ff_2_tanh = torch_nn.Tanh()
        self.l_ff_3 = torch_nn.Linear(hidden_size // 4, signal_size, bias=False)
        self.l_ff_3_tanh = torch_nn.Tanh()

        # a simple scale
        self.scale = torch_nn.Parameter(
            torch.tensor([1 / len(self.l_convs)]), requires_grad=False
        )
        return

    def forward(self, signal, context):
        """
        Assume: signal (batchsize=1, length, signal_size)
                context (batchsize=1, length, hidden_size)
        Output: (batchsize=1, length, signal_size)
        """
        # expand dimension
        tmp_hidden = self.l_ff_1_tanh(self.l_ff_1(signal))

        # loop over dilated convs
        # output of a d-conv is input + context + d-conv(input)
        for l_conv in self.l_convs:
            tmp_hidden = tmp_hidden + l_conv(tmp_hidden) + context

        # to be consistent with legacy configuration in CURRENNT
        tmp_hidden = tmp_hidden * self.scale

        # compress the dimesion and skip-add
        tmp_hidden = self.l_ff_2_tanh(self.l_ff_2(tmp_hidden))
        tmp_hidden = self.l_ff_3_tanh(self.l_ff_3(tmp_hidden))
        output_signal = tmp_hidden + signal

        return output_signal


# Sine waveform generator
class SineGen(torch_nn.Module):
    """Definition of sine generator
    SineGen(samp_rate, harmonic_num = 0,
            sine_amp = 0.1, noise_std = 0.003,
            voiced_threshold = 0,
            flag_for_pulse=False)

    samp_rate: sampling rate in Hz
    harmonic_num: number of harmonic overtones (default 0)
    sine_amp: amplitude of sine-wavefrom (default 0.1)
    noise_std: std of Gaussian noise (default 0.003)
    voiced_thoreshold: F0 threshold for U/V classification (default 0)
    flag_for_pulse: this SinGen is used inside PulseGen (default False)

    Note: when flag_for_pulse is True, the first time step of a voiced
        segment is always sin(np.pi) or cos(0)
    """

    def __init__(
        self,
        samp_rate,
        harmonic_num=0,
        sine_amp=0.1,
        noise_std=0.003,
        voiced_threshold=0,
        flag_for_pulse=False,
    ):
        super(SineGen, self).__init__()
        self.sine_amp = sine_amp
        self.noise_std = noise_std
        self.harmonic_num = harmonic_num
        self.dim = self.harmonic_num + 1
        self.sampling_rate = samp_rate
        self.voiced_threshold = voiced_threshold
        self.flag_for_pulse = flag_for_pulse

    def _f02uv(self, f0):
        # generate uv signal
        uv = torch.ones_like(f0)
        uv = uv * (f0 > self.voiced_threshold)
        return uv

    def _f02sine(self, f0_values):
        """f0_values: (batchsize, length, dim)
        where dim indicates fundamental tone and overtones
        """
        # convert to F0 in rad. The interger part n can be ignored
        # because 2 * np.pi * n doesn't affect phase
        rad_values = (f0_values / self.sampling_rate) % 1

        # initial phase noise (no noise for fundamental component)
        rand_ini = torch.rand(
            f0_values.shape[0], f0_values.shape[2], device=f0_values.device
        )
        rand_ini[:, 0] = 0
        rad_values[:, 0, :] = rad_values[:, 0, :] + rand_ini

        # instantanouse phase sine[t] = sin(2*pi \sum_i=1 ^{t} rad)
        if not self.flag_for_pulse:
            # for normal case

            # To prevent torch.cumsum numerical overflow,
            # it is necessary to add -1 whenever \sum_k=1^n rad_value_k > 1.
            # Buffer tmp_over_one_idx indicates the time step to add -1.
            # This will not change F0 of sine because (x-1) * 2*pi = x *2*pi
            tmp_over_one = torch.cumsum(rad_values, 1) % 1
            tmp_over_one_idx = (tmp_over_one[:, 1:, :] - tmp_over_one[:, :-1, :]) < 0
            cumsum_shift = torch.zeros_like(rad_values)
            cumsum_shift[:, 1:, :] = tmp_over_one_idx * -1.0

            sines = torch.sin(
                torch.cumsum(rad_values + cumsum_shift, dim=1) * 2 * np.pi
            )
        else:
            # If necessary, make sure that the first time step of every
            # voiced segments is sin(pi) or cos(0)
            # This is used for pulse-train generation

            # identify the last time step in unvoiced segments
            uv = self._f02uv(f0_values)
            uv_1 = torch.roll(uv, shifts=-1, dims=1)
            uv_1[:, -1, :] = 1
            u_loc = (uv < 1) * (uv_1 > 0)

            # get the instantanouse phase
            tmp_cumsum = torch.cumsum(rad_values, dim=1)
            # different batch needs to be processed differently
            for idx in range(f0_values.shape[0]):
                temp_sum = tmp_cumsum[idx, u_loc[idx, :, 0], :]
                temp_sum[1:, :] = temp_sum[1:, :] - temp_sum[0:-1, :]
                # stores the accumulation of i.phase within
                # each voiced segments
                tmp_cumsum[idx, :, :] = 0
                tmp_cumsum[idx, u_loc[idx, :, 0], :] = temp_sum

            # rad_values - tmp_cumsum: remove the accumulation of i.phase
            # within the previous voiced segment.
            i_phase = torch.cumsum(rad_values - tmp_cumsum, dim=1)

            # get the sines
            sines = torch.cos(i_phase * 2 * np.pi)
        return sines

    def forward(self, f0):
        """sine_tensor, uv = forward(f0)
        input F0: tensor(batchsize=1, length, dim=1)
                  f0 for unvoiced steps should be 0
        output sine_tensor: tensor(batchsize=1, length, dim)
        output uv: tensor(batchsize=1, length, 1)
        """
        with torch.no_grad():
            f0_buf = torch.zeros(f0.shape[0], f0.shape[1], self.dim, device=f0.device)
            # fundamental component
            f0_buf[:, :, 0] = f0[:, :, 0]
            for idx in np.arange(self.harmonic_num):
                # idx + 2: the (idx+1)-th overtone, (idx+2)-th harmonic
                f0_buf[:, :, idx + 1] = f0_buf[:, :, 0] * (idx + 2)

            # generate sine waveforms
            sine_waves = self._f02sine(f0_buf) * self.sine_amp

            # generate uv signal
            # uv = torch.ones(f0.shape)
            # uv = uv * (f0 > self.voiced_threshold)
            uv = self._f02uv(f0)

            # noise: for unvoiced should be similar to sine_amp
            #        std = self.sine_amp/3 -> max value ~ self.sine_amp
            # .       for voiced regions is self.noise_std
            noise_amp = uv * self.noise_std + (1 - uv) * self.sine_amp / 3
            noise = noise_amp * torch.randn_like(sine_waves)

            # first: set the unvoiced part to 0 by uv
            # then: additive noise
            sine_waves = sine_waves * uv + noise
        return sine_waves, uv, noise


## For condition module only provide Spectral feature to Filter block
class CondModuleHnSincNSF(torch_nn.Module):
    """Condition module for hn-sinc-NSF

    Upsample and transform input features
    CondModuleHnSincNSF(input_dimension, output_dimension, up_sample_rate,
               blstm_dimension = 64, cnn_kernel_size = 3)

    Spec, F0, cut_off_freq = CondModuleHnSincNSF(features, F0)

    Both input features should be frame-level features
    If x doesn't contain F0, just ignore the returned F0

    CondModuleHnSincNSF(input_dim, output_dim, up_sample,
                        blstm_s = 64, cnn_kernel_s = 3,
                        voiced_threshold = 0):

    input_dim: sum of dimensions of input features
    output_dim: dim of the feature Spec to be used by neural filter-block
    up_sample: up sampling rate of input features
    blstm_s: dimension of the features from blstm (default 64)
    cnn_kernel_s: kernel size of CNN in condition module (default 3)
    voiced_threshold: f0 > voiced_threshold is voiced, otherwise unvoiced
    """

    def __init__(
        self,
        input_dim,
        output_dim,
        up_sample,
        blstm_s=64,
        cnn_kernel_s=3,
        voiced_threshold=0,
        out_lf0_idx=60,
        upsample_net=None,
        upsample_params={},
        concat_norm_f0=True,
    ):
        super(CondModuleHnSincNSF, self).__init__()

        # input feature dimension
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.up_sample = up_sample
        self.blstm_s = blstm_s
        self.cnn_kernel_s = cnn_kernel_s
        self.cut_f_smooth = up_sample * 4
        self.voiced_threshold = voiced_threshold
        self.out_lf0_idx = out_lf0_idx
        self.aux_context_window = upsample_params.get("aux_context_window", 0)
        self.concat_norm_f0 = concat_norm_f0

        # Use PWG style upsampling
        if upsample_net is not None:
            self.pwg_style_upsample = True
            self.upsample_net = getattr(upsample, upsample_net)(**upsample_params)
            l_conv1d_in_dim = input_dim
        # Original NSF-style upsampling
        else:
            self.pwg_style_upsample = False
            self.l_blstm = BLSTMLayer(input_dim, self.blstm_s)
            l_conv1d_in_dim = self.blstm_s
            # Upsampling layer for hidden features
            self.l_upsamp = UpSampleLayer(self.output_dim, self.up_sample, True)

        # the CNN layer (+1 dim for cut_off_frequence of sinc filter)
        self.l_conv1d = Conv1dKeepLength(
            l_conv1d_in_dim, self.output_dim, dilation_s=1, kernel_s=self.cnn_kernel_s
        )

        # separate layer for up-sampling normalized F0 values
        self.l_upsamp_f0_hi = UpSampleLayer(1, self.up_sample, True)

        # Upsampling for F0: don't smooth up-sampled F0
        self.l_upsamp_F0 = UpSampleLayer(1, self.up_sample, False)

        # Another smoothing layer to smooth the cut-off frequency
        # for sinc filters. Use a larger window to smooth
        self.l_cut_f_smooth = MovingAverage(1, self.cut_f_smooth)

    def get_cut_f(self, hidden_feat, f0):
        """cut_f = get_cut_f(self, feature, f0)
        feature: (batchsize, length, dim=1)
        f0: (batchsize, length, dim=1)
        """
        # generate uv signal
        uv = torch.ones_like(f0) * (f0 > self.voiced_threshold)
        # hidden_feat is between (-1, 1) after conv1d with tanh
        # (-0.2, 0.2) + 0.3 = (0.1, 0.5)
        # voiced:   (0.1, 0.5) + 0.4 = (0.5, 0.9)
        # unvoiced: (0.1, 0.5) = (0.1, 0.5)
        return hidden_feat * 0.2 + uv * 0.4 + 0.3

    def forward(self, feature, f0):
        """spec, f0 = forward(self, feature, f0)
        feature: (batchsize, length, dim)
        f0: (batchsize, length, dim=1), which should be F0 at frame-level

        spec: (batchsize, length, self.output_dim), at wave-level
        f0: (batchsize, length, 1), at wave-level
        """
        if self.pwg_style_upsample:
            tmp = self.l_conv1d(
                self.upsample_net(feature.transpose(1, 2)).transpose(1, 2)
            )
        else:
            tmp = self.l_upsamp(self.l_conv1d(self.l_blstm(feature)))

        # concatenat normed F0 with hidden spectral features
        if self.concat_norm_f0:
            up_norm_f0 = self.l_upsamp_f0_hi(
                feature[:, :, self.out_lf0_idx : self.out_lf0_idx + 1]
            )
            if self.aux_context_window > 0:
                w = self.aux_context_window * self.up_sample
                up_norm_f0 = up_norm_f0[:, w:-w, :]
            context = torch.cat(
                (
                    tmp[:, :, 0 : self.output_dim - 1],
                    up_norm_f0,
                ),
                dim=2,
            )
        else:
            context = tmp

        # hidden feature for cut-off frequency
        hidden_cut_f = tmp[:, :, self.output_dim - 1 :]

        # directly up-sample F0 without smoothing
        f0_upsamp = self.l_upsamp_F0(f0)

        # get the cut-off-frequency from output of CNN
        cut_f = self.get_cut_f(hidden_cut_f, f0_upsamp)
        # smooth the cut-off-frequency using fixed average smoothing
        cut_f_smoothed = self.l_cut_f_smooth(cut_f)

        # return
        return context, f0_upsamp, cut_f_smoothed, hidden_cut_f


# For source module
class SourceModuleHnNSF(torch_nn.Module):
    """SourceModule for hn-nsf
    SourceModule(sampling_rate, harmonic_num=0, sine_amp=0.1,
                 add_noise_std=0.003, voiced_threshod=0)
    sampling_rate: sampling_rate in Hz
    harmonic_num: number of harmonic above F0 (default: 0)
    sine_amp: amplitude of sine source signal (default: 0.1)
    add_noise_std: std of additive Gaussian noise (default: 0.003)
        note that amplitude of noise in unvoiced is decided
        by sine_amp
    voiced_threshold: threhold to set U/V given F0 (default: 0)

    Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
    F0_sampled (batchsize, length, 1)
    Sine_source (batchsize, length, 1)
    noise_source (batchsize, length 1)
    uv (batchsize, length, 1)
    """

    def __init__(
        self,
        sampling_rate,
        harmonic_num=0,
        sine_amp=0.1,
        add_noise_std=0.003,
        voiced_threshod=0,
    ):
        super(SourceModuleHnNSF, self).__init__()

        self.sine_amp = sine_amp
        self.noise_std = add_noise_std

        # to produce sine waveforms
        self.l_sin_gen = SineGen(
            sampling_rate, harmonic_num, sine_amp, add_noise_std, voiced_threshod
        )

        # to merge source harmonics into a single excitation
        self.l_linear = torch_nn.Linear(harmonic_num + 1, 1)
        self.l_tanh = torch_nn.Tanh()

    def forward(self, x):
        """
        Sine_source, noise_source = SourceModuleHnNSF(F0_sampled)
        F0_sampled (batchsize, length, 1)
        Sine_source (batchsize, length, 1)
        noise_source (batchsize, length 1)
        """
        # source for harmonic branch
        sine_wavs, uv, _ = self.l_sin_gen(x)
        sine_merge = self.l_tanh(self.l_linear(sine_wavs))

        # source for noise branch, in the same shape as uv
        noise = torch.randn_like(uv) * self.sine_amp / 3
        return sine_merge, noise, uv


# For Filter module
class FilterModuleHnSincNSF(torch_nn.Module):
    """Filter for Hn-sinc-NSF
    FilterModuleHnSincNSF(signal_size, hidden_size, sinc_order = 31,
                          block_num = 5, kernel_size = 3,
                          conv_num_in_block = 10)
    signal_size: signal dimension (should be 1)
    hidden_size: dimension of hidden features inside neural filter block
    sinc_order: order of the sinc filter
    block_num: number of neural filter blocks in harmonic branch
    kernel_size: kernel size in dilated CNN
    conv_num_in_block: number of d-conv1d in one neural filter block

    Usage:
    output = FilterModuleHnSincNSF(har_source, noi_source, cut_f, context)
    har_source: source for harmonic branch (batchsize, length, dim=1)
    noi_source: source for noise branch (batchsize, length, dim=1)
    cut_f: cut-off-frequency of sinc filters (batchsize, length, dim=1)
    context: hidden features to be added (batchsize, length, dim)
    output: (batchsize, length, dim=1)
    """

    def __init__(
        self,
        signal_size,
        hidden_size,
        sinc_order=31,
        block_num=5,
        kernel_size=3,
        conv_num_in_block=10,
    ):
        super(FilterModuleHnSincNSF, self).__init__()
        self.signal_size = signal_size
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.block_num = block_num
        self.conv_num_in_block = conv_num_in_block
        self.sinc_order = sinc_order

        # filter blocks for harmonic branch
        tmp = [
            NeuralFilterBlock(signal_size, hidden_size, kernel_size, conv_num_in_block)
            for x in range(self.block_num)
        ]
        self.l_har_blocks = torch_nn.ModuleList(tmp)

        # filter blocks for noise branch (only one block, 5 sub-blocks)
        tmp = [
            NeuralFilterBlock(
                signal_size, hidden_size, kernel_size, conv_num_in_block // 2
            )
            for x in range(1)
        ]
        self.l_noi_blocks = torch_nn.ModuleList(tmp)

        # sinc filter generators and time-variant filtering layer
        self.l_sinc_coef = SincFilter(self.sinc_order)
        self.l_tv_filtering = TimeVarFIRFilter()
        # done

    def forward(self, har_component, noi_component, cond_feat, cut_f):
        # harmonic component
        for l_har_block in self.l_har_blocks:
            har_component = l_har_block(har_component, cond_feat)
        # noise componebt
        for l_noi_block in self.l_noi_blocks:
            noi_component = l_noi_block(noi_component, cond_feat)

        # get sinc filter coefficients
        lp_coef, hp_coef = self.l_sinc_coef(cut_f)

        # time-variant filtering
        har_signal = self.l_tv_filtering(har_component, lp_coef)
        noi_signal = self.l_tv_filtering(noi_component, hp_coef)

        # get output
        return har_signal + noi_signal


class HnSincNSF(torch_nn.Module):
    """Model definition"""

    def __init__(
        self,
        cin_channels,
        out_channels,
        upsample_rate=120,
        sample_rate=16000,
        out_lf0_idx=60,
        out_lf0_mean=5.953093881972361,
        out_lf0_scale=0.23435173188961034,
        out_vuv_idx=61,
        out_vuv_mean=0.8027627522670242,
        out_vuv_scale=0.39791295007789834,
        sine_amp=0.1,
        noise_std=0.003,
        hidden_dim=64,
        cnn_kernel_s=3,
        filter_block_num=5,
        cnn_num_in_block=10,
        harmonic_num=7,
        sinc_order=31,
        vuv_threshold=0.3,
        aux_context_window=0,
        upsample_net=None,
        upsample_params=None,
        concat_norm_f0=True,
    ):
        super(HnSincNSF, self).__init__()

        self.in_dim = cin_channels
        self.out_dim = out_channels
        self.out_lf0_idx = out_lf0_idx
        self.out_lf0_mean = out_lf0_mean
        self.out_lf0_scale = out_lf0_scale
        self.out_vuv_idx = out_vuv_idx
        self.out_vuv_mean = out_vuv_mean
        self.out_vuv_scale = out_vuv_scale
        self.vuv_threshold = vuv_threshold
        self.aux_context_window = aux_context_window

        # configurations
        # amplitude of sine waveform (for each harmonic)
        self.sine_amp = sine_amp
        # standard deviation of Gaussian noise for additive noise
        self.noise_std = noise_std
        # dimension of hidden features in filter blocks
        self.hidden_dim = hidden_dim
        # upsampling rate on input acoustic features (16kHz * 5ms = 80)
        # assume input_reso has the same value
        self.upsamp_rate = upsample_rate
        # sampling rate (Hz)
        self.sampling_rate = sample_rate
        # CNN kernel size in filter blocks
        self.cnn_kernel_s = cnn_kernel_s
        # number of filter blocks (for harmonic branch)
        # noise branch only uses 1 block
        self.filter_block_num = filter_block_num
        # number of dilated CNN in each filter block
        self.cnn_num_in_block = cnn_num_in_block
        # number of harmonic overtones in source
        self.harmonic_num = harmonic_num
        # order of sinc-windowed-FIR-filter
        self.sinc_order = sinc_order

        # Upsampling network configuration
        # NOTE: NSF-style upmpsaling used if `upsample_net` is None.
        # otherwise PWG-style upsampling is used.
        if upsample_params is not None:
            upsample_params.update(
                {"aux_channels": self.in_dim, "aux_context_window": aux_context_window}
            )
        else:
            upsample_params = {}

        # the three modules
        self.m_cond = CondModuleHnSincNSF(
            self.in_dim,
            self.hidden_dim,
            self.upsamp_rate,
            cnn_kernel_s=self.cnn_kernel_s,
            out_lf0_idx=out_lf0_idx,
            upsample_net=upsample_net,
            upsample_params=upsample_params,
            concat_norm_f0=concat_norm_f0,
        )

        self.m_source = SourceModuleHnNSF(
            self.sampling_rate, self.harmonic_num, self.sine_amp, self.noise_std
        )

        self.m_filter = FilterModuleHnSincNSF(
            self.out_dim,
            self.hidden_dim,
            self.sinc_order,
            self.filter_block_num,
            self.cnn_kernel_s,
            self.cnn_num_in_block,
        )

    def forward(self, x):
        """definition of forward method
        Assume x (batchsize=1, length, dim)
        Return output(batchsize=1, length)
        """
        # (B, C, T) -> (B, T, C)
        x = x.transpose(1, 2)
        lf0 = (
            x[:, :, self.out_lf0_idx].unsqueeze(-1) * self.out_lf0_scale
            + self.out_lf0_mean
        )
        vuv = (
            x[:, :, self.out_vuv_idx].unsqueeze(-1) * self.out_vuv_scale
            + self.out_vuv_mean
        )

        if self.aux_context_window > 0:
            lf0 = lf0[:, self.aux_context_window : -self.aux_context_window]
            vuv = vuv[:, self.aux_context_window : -self.aux_context_window]

        f0 = torch.exp(lf0)
        f0[vuv < self.vuv_threshold] = 0

        # condition module
        # feature-to-filter-block, f0-up-sampled, cut-off-f-for-sinc,
        #  hidden-feature-for-cut-off-f
        cond_feat, f0_upsamped, cut_f, _ = self.m_cond(x, f0)

        # source module
        # harmonic-source, noise-source (for noise branch), uv
        har_source, noi_source, _ = self.m_source(f0_upsamped)

        # neural filter module (including sinc-based FIR filtering)
        # output
        output = self.m_filter(har_source, noi_source, cond_feat, cut_f)

        # (B, T, C) -> (B, C, T)
        return output.transpose(1, 2)

    def inference(self, c=None, x=None, normalize_before=False):
        if not isinstance(c, torch.Tensor):
            c = torch.tensor(c, dtype=torch.float).to(next(self.parameters()).device)
        if normalize_before:
            assert False
        c = self.forward(c.transpose(1, 0).unsqueeze(0))
        return c.squeeze(0).transpose(1, 0)

    def remove_weight_norm(self):
        """Remove weight normalization module from all of the layers."""

        def _remove_weight_norm(m):
            try:
                torch.nn.utils.remove_weight_norm(m)
            except ValueError:  # this module didn't have weight norm
                return

        self.apply(_remove_weight_norm)

    def apply_weight_norm(self):
        """Apply weight normalization module from all of the layers."""

        def _apply_weight_norm(m):
            if isinstance(m, torch.nn.Conv1d) or isinstance(m, torch.nn.Conv2d):
                torch.nn.utils.weight_norm(m)

        self.apply(_apply_weight_norm)
