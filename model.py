import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras.layers import Layer
from dgl.utils import expand_as_pair


class SAGEConv(Layer):
    """
    """

    def __init__(self,
                 in_dim,
                 out_dim,
                 aggregator_type,
                 bias=True,
                 norm=None,
                 activation=None):
        """
        `in_dim` : input feature size

        `out_dim` : output feature size

        `aggregator_type` : aggregator function type

        `bias` : If to add a bias

        `norm` : feature normalization

        `activation` : activation function
        """
        super(SAGEConv, self).__init__()

        self._in_src_dim, self._in_dst_dim = expand_as_pair(in_dim)
        self._out_dim = out_dim
        self._agg_type = aggregator_type
        self.bias = bias
        self.norm = norm
        self.activation = activation

        if self._agg_type not in ["mean", "max_pool", "lstm", "gcn"]:
            raise KeyError(
                'Aggregator type {} not supported.'.format(self._agg_type))
        if self._agg_type == "max_pool":
            self.fc_pool = tf.keras.models.Sequential()
            self.fc_pool.add(layers.Dense(self._in_src_dim,
                                          input_shape=(self._in_src_dim)))
        if self._agg_type == "lstm":
            self.lstm = layers.LSTM(units=self.in_src_dim)

        if self._agg_type in ["mean", "lstm", "max_pool"]:
            self.fc_self = tf.keras.models.Sequential()
            self.fc_self.add(layers.Dense(
                units=self._in_dst_dim, input_shape=(self._in_src_dim,), use_bias=bias))

        self.fc_neigh = self.fc_self = tf.keras.models.Sequential()
        self.fc_self.add(layers.Dense(
            units=self._in_dst_dim, input_shape=(self._in_src_dim,), use_bias=bias))
        self.reset()

    def reset(self):
        gain = nn.init.calculate_gain('relu')
        if self._aggre_type == 'max_pool':
            nn.init.xavier_uniform_(self.fc_pool.weight, gain=gain)
        if self._aggre_type == 'lstm':
            self.lstm.
        if self._aggre_type != 'gcn':
            nn.init.xavier_uniform_(self.fc_self.weight, gain=gain)
        nn.init.xavier_uniform_(self.fc_neigh.weight, gain=gain)
