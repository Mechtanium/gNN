# Copyright 2021 Predicitve Intelligence Lab
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# The Deep Galerkin Method reference architecture from jaxpinns (Predictive
# Intelligence Lab), kept verbatim: modules.models copies its Glorot
# initialisation arrays into the NNX module so the two are bit-identical, and
# build_pipeline asserts that parity at every build.

import jax.numpy as np
from jax import random

def DGM(layers, l=3):
    ''' Deep Galerkin Method architecture '''
    if len(layers) != 3:
        raise ValueError("DGM expects layers=[dim_d, M, dim_f].")
    if l < 1:
        raise ValueError("DGM expects l >= 1 recurrent blocks.")

    dim_d, M, dim_f = layers
    if dim_d < 1 or M < 1 or dim_f < 1:
        raise ValueError("DGM layer sizes must be positive integers.")

    def init(rng_key):
        def init_weight(key, d_in, d_out):
            glorot_stddev = 1. / np.sqrt((d_in + d_out) / 2.)
            return glorot_stddev * random.normal(key, (d_in, d_out))

        num_keys = 2 + 8*l
        keys = random.split(rng_key, num_keys)
        idx = 0

        # Input projection and final readout
        W1 = init_weight(keys[idx], dim_d, M)
        idx += 1
        Wout = init_weight(keys[idx], M, dim_f)
        idx += 1
        b1 = np.zeros(M)
        bout = np.zeros(dim_f)

        # Per-recurrent-block parameters
        block_params = []
        for _ in range(l):
            Uz = init_weight(keys[idx], dim_d, M)
            idx += 1
            Ug = init_weight(keys[idx], dim_d, M)
            idx += 1
            Ur = init_weight(keys[idx], dim_d, M)
            idx += 1
            Uh = init_weight(keys[idx], dim_d, M)
            idx += 1

            Wz = init_weight(keys[idx], M, M)
            idx += 1
            Wg = init_weight(keys[idx], M, M)
            idx += 1
            Wr = init_weight(keys[idx], M, M)
            idx += 1
            Wh = init_weight(keys[idx], M, M)
            idx += 1

            bz = np.zeros(M)
            bg = np.zeros(M)
            br = np.zeros(M)
            bh = np.zeros(M)

            block_params.append((Uz, Ug, Ur, Uh, Wz, Wg, Wr, Wh, bz, bg, br, bh))

        params = ((W1, b1), tuple(block_params), (Wout, bout))
        return params

    def apply(params, inputs):
        (W1, b1), block_params, (Wout, bout) = params
        X = np.tanh(np.dot(inputs, W1) + b1)

        for block in block_params:
            Uz, Ug, Ur, Uh, Wz, Wg, Wr, Wh, bz, bg, br, bh = block
            Z = np.tanh(np.dot(inputs, Uz) + np.dot(X, Wz) + bz)
            G = np.tanh(np.dot(inputs, Ug) + np.dot(X, Wg) + bg)
            R = np.tanh(np.dot(inputs, Ur) + np.dot(X, Wr) + br)
            H = np.tanh(np.dot(inputs, Uh) + np.dot(X*R, Wh) + bh)
            X = (1.0 - G) * H + Z * X

        outputs = np.dot(X, Wout) + bout
        return outputs

    return init, apply


