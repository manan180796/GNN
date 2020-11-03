import tensorflow as tf
from tensorflow.keras import layers
from spektral.layers import GraphConv


class GCN(tf.keras.Model):
    def __init__(self, n: int, embed: int = 1):
        super(GCN, self).__init__()
        self.embedding = layers.Embedding(n, embed,embeddings_initializer="glorot_uniform")
        self.conv1 = GraphConv(3, "relu")
        self.conv2 = GraphConv(3, "relu")
        self.conv3 = GraphConv(3, "relu")
        # self.drop = 
        # self.conv2 = GraphConv(2, "relu")
        self.final = layers.Dense(2, "relu")

    # @tf.function
    def call(self, inputs):
        graph, feature = inputs
        feature = self.embedding(feature)
        # print(graph)
        feature = self.conv1([feature, graph])
        feature = self.conv2([feature, graph])
        feature = self.conv3([feature, graph])
        # feature = self.conv2([feature, graph])
        return self.final(feature)
        # return feature

    def train_step(self, data):
        # Unpack the data. Its structure depends on your model and
        # on what you pass to `fit()`.
        if len(data) == 3:
            x, y, sample_weight = data
        else:
            x, y = data

        with tf.GradientTape() as tape:
            y_pred = self(x, training=True)  # Forward pass
            # tf.print(y_pred)
            # Compute the loss value.
            # The loss function is configured in `compile()`.
            loss = self.compiled_loss(
                y,
                y_pred,
                sample_weight=sample_weight,
                regularization_losses=self.losses,
            )

        # Compute gradients
        trainable_vars = self.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)
        # tf.print(gradients)
        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))
        # Update metrics (includes the metric that tracks the loss)
        self.compiled_metrics.update_state(y, y_pred)
        # Return a dict mapping metric names to current value
        return {m.name: m.result() for m in self.metrics}
