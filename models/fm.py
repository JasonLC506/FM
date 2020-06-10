"""
factorization machine rank-2
with fixed maximum none-zero feature dimensions
for multi-class classification
"""
import tensorflow as tf

from models import NN


Optimizer = tf.train.AdamOptimizer


class FM(NN):
    def __init__(
            self,
            feature_shape,
            feature_dim,
            label_dim,
            model_spec,
            model_name=None
    ):
        self.feature_shape = feature_shape                # total feature dimension, final (-1) as paddle
        self.feature_dim = feature_dim                    # maximum none-zero feature dimensions
        self.label_dim = label_dim                        # multi-class classification
        self.model_spec = model_spec
        if model_name is None:
            self.model_name = model_spec["name"]
        else:
            self.model_name = model_name

        super(FM, self).__init__(graph=None)

    def initialization(self):
        self.sess = self.setup_session(graph=self.graph)
        self.initialize(sess=self.sess)

    def _setup_placeholder(self):
        with tf.name_scope("placeholder"):
            self.feature_index = tf.placeholder(
                dtype=tf.int64,
                shape=[None, self.feature_dim],
                name="feature_index"
            )
            self.feature_value = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.feature_dim],
                name="feature_value"
            )
            self.label = tf.placeholder(
                dtype=tf.float32,
                shape=[None, self.label_dim],
                name="label"
            )

    def _setup_net(self):
        # rank-0 #
        self.bias = tf.Variable(
            tf.zeros(
                [1, self.label_dim],
                dtype=tf.float32
            ),
            dtype=tf.float32,
            name="bias",
            trainable=self.model_spec["trainable_bias"]
        )

        # rank-1 #
        self.weights = tf.Variable(
            tf.random_normal(
                [self.feature_shape, self.label_dim],
                mean=0.0,
                stddev=self.model_spec["stddev_weights"]
            ),
            dtype=tf.float32,
            name="weights",
            trainable=self.model_spec["trainable_weights"]
        )
        feature_weight = tf.nn.embedding_lookup(self.weights, self.feature_index)
        linear = tf.einsum(
            "ijk,ij->ik",
            feature_weight,
            self.feature_value
        )

        # rank-2 #
        embedding_dim = self.model_spec["embedding_dim"]
        self.embeddings = tf.Variable(
            tf.random_normal(
                [self.feature_shape, embedding_dim * self.label_dim],
                mean=0.0,
                stddev=0.01
            ),
            dtype=tf.float32,
            name='embeddings'
        )

        feature_emb = tf.nn.embedding_lookup(self.embeddings, self.feature_index)
        feature_emb_scaled = feature_emb * tf.expand_dims(self.feature_value, axis=-1)
        feature_emb_scaled = tf.reshape(feature_emb_scaled, shape=[-1, self.feature_dim, embedding_dim, self.label_dim])
        bilinear_a = tf.reduce_sum(
            feature_emb_scaled,
            axis=1
        )
        bilinear_a = tf.pow(
            tf.norm(
                bilinear_a,
                ord=2,
                axis=1
            ),
            2.0
        )
        bilinear_b = tf.pow(
            tf.norm(
                feature_emb_scaled,
                ord='fro',
                axis=[1, 2]
            ),
            2.0
        )
        bilinear = (bilinear_a - bilinear_b) / 2.0

        # sum together #
        self.logits = bilinear + linear + self.bias
        self.preds = tf.nn.softmax(self.logits)

    def _setup_loss(self):
        cross_entropy = tf.nn.softmax_cross_entropy_with_logits(
            labels=self.label,
            logits=self.logits,
            name="cross_entropy"
        )
        self.loss_cross_entropy = tf.reduce_sum(cross_entropy)
        self.loss = self.loss_cross_entropy

    def _setup_optim(self):
        self.optimizer = Optimizer(
            learning_rate=self.model_spec["learning_rate"],
            epsilon=1e-06,
            name="optimizer"
        ).minimize(self.loss)

    def train(
            self,
            data_generator,
            data_generator_valid=None
    ):
        if self.sess is None:
            self.initialization()
        results = self._train_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_train,
            op_optimizer=self.optimizer,
            op_losses=[self.loss],
            session=self.sess,
            batch_size=self.model_spec["batch_size"],
            max_epoch=self.model_spec["max_epoch"],
            data_generator_valid=data_generator_valid,
            op_savers=[self.saver],
            save_path_prefixs=[self.model_name],
            log_board_dir="../summary/" + self.model_name
        )
        return results

    def _fn_feed_dict_train(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.feature_index: data["feature_index"][batch_index],
            self.feature_value: data["feature_value"][batch_index],
            self.label: data['label'][batch_index]
        }
        return feed_dict

    def predict(
            self,
            data_generator
    ):
        results = self._feed_forward_w_generator(
            data_generator=data_generator,
            fn_feed_dict=self._fn_feed_dict_predict,
            output=[self.preds],
            session=self.sess,
            batch_size=self.model_spec["batch_size"]
        )[0]
        return results

    def _fn_feed_dict_predict(
            self,
            data,
            batch_index
    ):
        feed_dict = {
            self.feature_index: data["feature_index"][batch_index],
            self.feature_value: data["feature_value"][batch_index]
        }
        return feed_dict
