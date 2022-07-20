import multiprocessing as mp
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow_addons.activations import sparsemax


def GLU(x):
    return x * tf.sigmoid(x)

class FCBlock(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.layer = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x):
        return GLU(self.bn(self.layer(x)))


class SharedBlock(tf.keras.layers.Layer):
    def __init__(self, units, mult=tf.sqrt(0.5)):
        super().__init__()
        self.layer1 = FCBlock(units)
        self.layer2 = FCBlock(units)
        self.mult = mult

    def call(self, x):
        out1 = self.layer1(x)
        out2 = self.layer2(out1)
        return out2 + self.mult * out1

class DecisionBlock(SharedBlock):
    def __init__(self, units, mult=tf.sqrt(0.5)):
        super().__init__(units, mult)

    def call(self, x):
        out1 = x * self.mult + self.layer1(x)
        out2 = out1 * self.mult + self.layer2(out1)
        return out2

class Prior(tf.keras.layers.Layer):
    def __init__(self, gamma=1.1):
        super().__init__()
        self.gamma = gamma

    def reset(self):
        self.P = 1.0

    def call(self, mask):
        self.P = self.P * (self.gamma - mask)
        return self.P

class AttentiveTransformer(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        self.layer = tf.keras.layers.Dense(units)
        self.bn = tf.keras.layers.BatchNormalization()

    def call(self, x, prior):
        return sparsemax(prior * self.bn(self.layer(x)))

class TabNet(tf.keras.Model):
    def __init__(self, input_dim , output_dim, steps, n_d, n_a, gamma=1.3):
        super().__init__()
        # hyper-parameters
        self.n_d, self.n_a, self.steps = n_d, n_a, steps
        # input-normalisation
        self.bn = tf.keras.layers.BatchNormalization()
        # Feature Transformer
        self.shared = SharedBlock(n_d+n_a)
        self.first_block = DecisionBlock(n_d+n_a)
        self.decision_blocks = [DecisionBlock(n_d+n_a)] * steps
        # Attentive Transformer
        self.attention = [AttentiveTransformer(input_dim)] * steps
        self.prior_scale = Prior(gamma)
        # final layer
        self.final = tf.keras.layers.Dense(output_dim)

        self.eps = 1e-8
        self.add_layer = tf.keras.layers.Add()

    @tf.function
    def call(self, x):
        self.prior_scale.reset()
        final_outs = []
        mask_losses = []

        x = self.bn(x)
        attention = self.first_block(self.shared(x))[:,:self.n_a]
        for i in range(self.steps):
            mask = self.attention[i](attention, self.prior_scale.P)
            entropy = mask * tf.math.log(mask + self.eps)
            mask_losses.append(
                -tf.reduce_sum(entropy, axis=-1) / self.steps
            )

            prior = self.prior_scale(mask)
            out = self.decision_blocks[i](self.shared(x * prior))
            attention, output = out[:,:self.n_a], out[:,self.n_a:]
            final_outs.append(tf.nn.relu(output))

        final_out = self.add_layer(final_outs)
        mask_loss = self.add_layer(mask_losses)

        return self.final(final_out), mask_loss

    def mask_importance(self, x):
        self.prior_scale.reset()
        feature_importance = 0

        x = self.bn(x)
        attention = self.first_block(self.shared(x))[:,:self.n_a]
        for i in range(self.steps):
            mask = self.attention[i](attention, self.prior_scale.P)

            prior = self.prior_scale(mask)
            out = self.decision_blocks[i](self.shared(x * prior))
            attention, output = out[:,:self.n_a], out[:,self.n_a:]
            step_importance = tf.reduce_sum(tf.nn.relu(output), axis=1, keepdims=True)
            feature_importance += mask * step_importance

        return feature_importance