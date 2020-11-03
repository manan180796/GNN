from tensorflow.keras.layers import Input, Dropout
from tensorflow.keras.models import Model
from spektral.layers import GraphConv
from spektral.datasets import citation

adj, x, y, train_mask, val_mask, test_mask = citation.load_data("cora")


N = adj.shape[0]
F = x.shape[-1]
n_classes = y.shape[-1]

X_in = Input(shape=(F, ))
A_in = Input((N, ), sparse=True)

X_1 = GraphConv(16, "relu")([X_in, A_in])
X_1 = Dropout(0.5)(X_1)
X_2 = GraphConv(n_classes, "softmax")([X_1, A_in])

model = Model(inputs=[X_in, A_in], outputs=X_2)
adj = GraphConv.preprocess(adj).astype("f4")
model.compile(optimizer="adam",
              loss="categorical_crossentropy",
              weighted_metrics=["acc"])
model.summary()
# Prepare data
x = x.toarray()
adj = adj.astype("f4")
validation_data = ([x, adj], y, val_mask)

# Train model
model.fit([x, adj], y,
          sample_weight=train_mask,
          validation_data=validation_data,
          batch_size=N,
          shuffle=False, epochs=100)
