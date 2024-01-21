import os
import logging
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
logging.getLogger('tensorflow').setLevel(logging.FATAL)
os.environ["KMP_AFFINITY"] = "noverbose"
from datetime import datetime
import numpy as np
import pandas as pd


from sklearn.utils import class_weight
from sklearn import cluster
import tensorflow as tf
from tensorflow.keras import losses, layers, models, callbacks, backend, \
    initializers, regularizers

from spektral.data import Graph, Dataset, PackedBatchLoader
from spektral.layers import GCNConv, GlobalSumPool, GlobalMaxPool, GlobalAvgPool, GraphMasking
from spektral.transforms import LayerPreprocess
np.random.seed(1)
tf.random.set_seed(2)
tf.autograph.set_verbosity(3)
tf.get_logger().setLevel('FATAL')

DATA_PATH = './testcases'
SAVE_PATH = './weights'
params = {
    'classes' : 1,
    'col_name' : 'rating',
    'obj_num' : 1,
    # start: 0,
    'norm' : True,
    'TIMEIT': 1,
}

class MSPDMixedNet(Dataset):
    def __init__(self, net_df, N, filtered_src_df, mode, graph_df, col_name,
                 obj_num, classes, TIMEIT,
                 **kwargs):  # replace/immitate src df later
        self.EPSILON = np.float32(0.0000001)
        self.objective = f'{col_name}{obj_num if obj_num == 3 else 2}{"abs" if col_name == "rating" else ""}'  # replace with err etc
        self.N = N
        self.amount_subgraphs = len(filtered_src_df)
        self.srcDf = filtered_src_df
        self.TIMEIT = TIMEIT
        self.classes = classes

        self.train = mode == 'train'
        if not self.train:
            inp = np.array(net_df.iloc[0], dtype=np.int32)
            self.data = np.array(inp[1:])
        else:
            self.net_idx = net_df[
                0]  # pass inputDf as np.array already with inputDf.iterrows()
            self.data = net_df[1:]
            self.graph_df = graph_df
            norm_row = self.graph_df[self.graph_df['sourceIdx'] == 0]

            self.objnum = 3

            s = np.array( self.graph_df[f'skew{self.objnum}'], dtype=np.float32)
            self.s_min = np.min(s)
            self.s_max = np.max(s) - self.s_min
            w = np.array( self.graph_df[f'wireLength{self.objnum}'], dtype=np.float32)
            self.w_min = np.min(w)
            self.w_max = np.max(w) - self.w_min
            self.norm_weights = np.array(
                norm_row[[f'wireLength{obj_num}', f'skew{obj_num}']],
                dtype=np.float32).reshape(2, )

        self.net = []
        super().__init__(**kwargs)

    def calc_y(self, src_idx_enum):

        src_row = self.graph_df[self.graph_df['sourceIdx'] == src_idx_enum]
        s = (src_row[f'skew{self.objnum}'] - self.s_min) / self.s_max
        w = (src_row[f'wireLength{self.objnum}'] - self.w_min) / self.w_max
#        return (1-w)

        x = np.array(src_row[[self.objective]], dtype=np.float32).reshape(1, )
        return x / 100
        # MIN_VAL = 50
        # x = (x-MIN_VAL)/ (100-MIN_VAL)
        #
        # return np.where(x > 0 , x , 0)

    def read(self) -> [Graph]:
        if self.TIMEIT:
            last_time = datetime.now()
        if self.TIMEIT == 1:
            _pre = 0
            _while = 0
            _post1 = 0
            _post2 = 0

        # xs, ys, center_dist, root_dist, quarter_class (5, fixed)
        AMOUNT_NODE_FEATS = 13

        # calc identical node features once for each subgraph
        xs = (self.data[::2] - 1) / 1001 - 0.5
        ys = (self.data[1::2] - 1) / 1001 - 0.5

        if xs[0] > 0:
            xs = -xs

        if ys[0] > 0:
            ys = -ys

        if ys[0] > xs[0]:
            xs, ys = ys, xs

        xs += 0.5
        ys += 0.5

        avg_x = np.average(xs)
        avg_y = np.average(ys)

        diff_com_xs = xs - avg_x
        diff_com_ys = ys - avg_y

        vert_half = np.array(diff_com_xs > 0, dtype=np.float32)
        com_dist = (np.abs(diff_com_xs) + np.abs(diff_com_ys))
        root_dist = (np.abs(xs - xs[0]) + np.abs(ys - ys[0]))
        denom = diff_com_xs
        for zero_i in np.where(denom == 0)[0]:
            denom[zero_i] = self.EPSILON
        ratio = diff_com_ys / denom
        abs_ratio = np.abs(ratio)
        pos_ratio = np.array(ratio >= 0, dtype=np.float32)
        gt1_ratio = np.array(abs_ratio >= 1, dtype=np.float32)
        lt1_ratio = 1 - gt1_ratio
        abs_ratio = lt1_ratio * abs_ratio + gt1_ratio * (
                    1 / (abs_ratio + self.EPSILON))
        abs_ratio -= 0.5

        node_feats = np.zeros((AMOUNT_NODE_FEATS, self.N), dtype=np.float32)
        node_feats[0] = xs
        node_feats[1] = ys
        node_feats[2] = com_dist / 2
        node_feats[3] = root_dist / 2
        node_feats[4] = abs_ratio
        node_feats[5] = pos_ratio
        node_feats[6] = gt1_ratio
        node_feats[7] = vert_half

        # Calc dist_mat once, copy and then modify for each src combination
        dist_mat = np.zeros((self.N, self.N), dtype=np.float32)

        for i in range(self.N):
            for j in range(i + 1, self.N):  # use symmetry of matrix
                dist = abs(xs[i] - xs[j]) + abs(ys[i] - ys[j])
                dist_mat[i][j] = dist
                dist_mat[j][i] = dist

        # dist_mat[:,0] = 0  # Remove Non-Root -> Root

        dist_mat /= 2

        node_feats = np.rollaxis(node_feats, -1)

        for _, row in self.srcDf.iterrows():
            if self.TIMEIT:
                last_time = datetime.now()

            A = np.copy(dist_mat)
            X = np.copy(node_feats)

            arr = np.array(row)
            src_idx_enum = arr[0]
            one_hot = arr[1:]

            for j in range(self.N):
                if one_hot[
                    j]:  # Remove non-root -> src (including src -> src (thus src <-> src))
                    # A[1:,j] = 0
                    pass


                else:  # Remove root -> non-src
                    A[0][j] = 0


            X[:, 8] = np.mean(A, axis=0)
            X[:, 9] = np.max(A, axis=0)
            X[:, 10] = np.mean(A, axis=1)
            X[:, 11] = np.max(A, axis=1)

            X[:, 12] = one_hot

            Y = self.calc_y(src_idx_enum)
            self.net.append(Graph(x=X, y=Y, a=A))

        return self.net

def make_clusters(xs, ys, k, linkage='single') -> [int]:
    # return np.arange(len(xs))
    k = min((len(xs) // 5) + 2, k)
    model = cluster.AgglomerativeClustering(n_clusters=k, linkage=linkage)
    cluster_model = model.fit([[x, y] for x, y in zip(xs, ys)])
    return cluster_model.labels_


class MSPDTrainSet(Dataset):
    CENTER_MID = 280
    BETWEEN_LOW = 360
    BETWEEN_MID = 495
    BETWEEN_HIGH = 640
    MID_CORNER = 780

    def __init__(self, N, start=0, end=None, TIMEIT=0, **kwargs):
        self.TIMEIT = TIMEIT

        self.data_obj_df = pd.read_csv(
            f"{DATA_PATH}/data_obj_stt_{N}.csv.gz", compression="gzip")
        self.input_df = pd.read_csv(f"{DATA_PATH}/input_stt_{N}.csv.gz",
                                    compression="gzip")
        self.src_df = pd.read_csv(f"{DATA_PATH}/sources_stt_{N}.csv.gz",
                                  compression="gzip")
        self.src_df = self.src_df[self.src_df['sourceIdx'] != 0]

        self.N = N
        self.amount_nets = len(self.input_df)
        self.obj_num = kwargs['obj_num']

        self.start = start
        self.end = self.amount_nets if end is None else end

        self.dataset = []
        super().__init__(**kwargs)

    def filter_sources(self, data) -> pd.DataFrame:

        filtered_df = self.src_df.copy()

        ###############################
        count = np.zeros(6, )
        xs = data[::2]
        ys = data[1::2]
        N = ys.shape[0]
        avg_x = 500
        avg_y = 500

        cluster_amount = 10

        root_x = xs[0]
        root_y = ys[0]

        root_dist_from_center = np.abs(root_x - 500) + np.abs(root_y - 500)

        root_dist_enum = -1

        if root_dist_from_center < self.CENTER_MID:
            count[0] += 1
            root_dist_enum = 0
        elif self.CENTER_MID <= root_dist_from_center <= self.BETWEEN_LOW:
            count[1] += 1
            root_dist_enum = 1
        elif self.BETWEEN_LOW <= root_dist_from_center <= self.BETWEEN_MID:
            count[2] += 1
            root_dist_enum = 2
        elif self.BETWEEN_MID <= root_dist_from_center <= self.BETWEEN_HIGH:
            count[3] += 1
            root_dist_enum = 3
        elif self.BETWEEN_HIGH <= root_dist_from_center <= self.MID_CORNER:
            count[4] += 1
            root_dist_enum = 4
        else:
            count[5] += 1
            root_dist_enum = 5

        diff_center_xs = xs - avg_x
        diff_center_ys = ys - avg_y

        # Rotating net to ensure root is in topleft quarter
        xs = xs - 501.5
        ys = ys - 501.5
        if xs[0] > 0: xs = -xs
        if ys[0] < 0: ys = -ys
        xs = xs + 501.5
        ys = ys + 501.5
        if ys[0] > -xs[0] + 1000:
            xs_new = 1000 - ys
            ys = 1000 - xs
            xs = xs_new

        ###### OBJECTIVE 1 #########
        if self.obj_num == 1:
            dist_threshold = [450, 500, 500, 550, 400, 600]
            x_center = [375, 350, 280, 180, 160, 300]
            y_center = [550, 450, 650, 675, 750, 725]
            pos = ["Center", "Between Center and Low", "Between Low and Mid",
                   "Between Mid and High", "Between High and Corner", "Corner"]
            include_1 = [False, True, True, True, True, True]
            include_2 = [True, True, True, True, True, False]
            include_3 = [True, True, True, True, True, False]

            while (True):

                new_vertices = [i for i, _ in enumerate(xs[1:], start=1)
                                if abs(xs[i] - x_center[root_dist_enum]) + abs(
                        ys[i] - y_center[root_dist_enum]) < dist_threshold[
                                    root_dist_enum]]
                new_xs = [xs[i] for i in new_vertices if i in new_vertices]
                new_ys = [ys[i] for i in new_vertices if i in new_vertices]
                if len(new_xs) > 4 or root_dist_enum == 5:
                    break
                dist_threshold[root_dist_enum] += 100

            new_dict = {i: idx for i, idx in enumerate(new_vertices)}
            ### clustering
            if not cluster_amount <= len(new_xs):
                cluster_amount = len(new_xs)
            if len(new_xs) > 1:
                clusterr = make_clusters(new_xs, new_ys, cluster_amount)
            else:
                clusterr = [0] if len(new_xs) == 1 else []

        ######### OBJECTIVE 2 ##########

        if self.obj_num == 2:
            A = np.zeros((self.N, self.N))
            for i in range(1, self.N):
                for j in range(i + 1, self.N):
                    dist = abs(xs[i] - xs[j]) + abs(ys[i] - ys[j])
                    A[i][j] = dist
                    A[j][i] = dist

            A[:, 0] = 0

            src_ids = [(i,) for i in range(1, self.N)]
            src_ids += [(i, j) for i in range(1, self.N)
                        for j in range(i + 1, self.N)
                        if 1300 > A[i][j] > 250.0]

            if self.N < 25:  # combinatorical reasons
                src_ids += [(i, j, k) for i in range(1, self.N)
                            for j in range(i + 1, self.N) for k in
                            range(j + 1, self.N)
                            if 1300 > A[i][j] > 200.0
                            and 1300 > A[i][k] > 200.0
                            and 1300 > A[j][k] > 200.0]

            mins = np.zeros(len(src_ids))
            maxs = np.zeros(len(src_ids))
            for src_id_enum, tup in enumerate(src_ids):
                mins[src_id_enum] = np.min([A[0][idx] for idx in tup])
                maxs[src_id_enum] = np.max(
                    np.min([A[idx] + A[0][idx] for idx in tup], axis=0)
                )

            min_skews = maxs - mins
            best_min_skew = min_skews[np.argmin(min_skews)]
            filtered_src_enum = \
                np.where(min_skews < best_min_skew * 1.15)[0]
            filtered_vertices = [src_ids[src_id_enum] for src_id_enum in
                                 filtered_src_enum]

        ###### OBJECTIVE 3 ######

        if self.obj_num == 3:
            dist_threshold = [450, 500, 500, 500, 400, 600]
            x_center = [350, 300, 280, 180, 160, 300]
            y_center = [550, 450, 650, 675, 750, 675]
            pos = ["Center", "Between Center and Low", "Between Low and Mid",
                   "Between Mid and High", "Between High and Corner", "Corner"]
            include_1 = [False, True, True, True, True, True]
            include_2 = [True, True, True, True, True, True]
            include_3 = [True, True, True, True, True, False]

            while (True):
                dist_threshold[root_dist_enum] += 100
                new_vertices = [i for i, _ in enumerate(xs[1:], start=1)
                                if abs(xs[i] - x_center[root_dist_enum]) + abs(
                        ys[i] - y_center[root_dist_enum]) < dist_threshold[
                                    root_dist_enum]]
                new_xs = [xs[i] for i in new_vertices if i in new_vertices]
                new_ys = [ys[i] for i in new_vertices if i in new_vertices]
                if len(new_xs) > 4:
                    break

            new_dict = {i: idx for i, idx in enumerate(new_vertices)}
            ### clustering
            if not cluster_amount <= len(new_xs):
                cluster_amount = len(new_xs)
            if len(new_xs) > 1:
                clusterr = make_clusters(new_xs, new_ys, cluster_amount)
            else:
                clusterr = [0] if len(new_xs) == 1 else []

        if self.obj_num != 2:
            ### filtering
            if include_1[root_dist_enum]:
                filtered_vertices_1 = [[i] for i, _ in enumerate(new_vertices)]
            else:
                filtered_vertices_1 = []
            if include_2[root_dist_enum]:
                filtered_vertices_2 = [[i, j] for i, _ in
                                       enumerate(new_vertices)
                                       for j, _ in
                                       enumerate(new_vertices[i + 1:],
                                                 start=i + 1)
                                       if clusterr[i] != clusterr[j]]
            else:
                filtered_vertices_2 = []
            if include_3[root_dist_enum]:
                filtered_vertices_3 = [[i, j, k] for i, _ in
                                       enumerate(new_vertices)
                                       for j, _ in
                                       enumerate(new_vertices[i + 1:],
                                                 start=i + 1)
                                       for k, _ in
                                       enumerate(new_vertices[j + 1:],
                                                 start=j + 1)
                                       if
                                       clusterr[i] != clusterr[j] and clusterr[
                                           i] != clusterr[k] and clusterr[j] !=
                                       clusterr[k]]
            else:
                filtered_vertices_3 = []

            diff_center_xs = xs - avg_x
            diff_center_ys = ys - avg_y
            vertex_dist_from_center = np.abs(diff_center_xs) + np.abs(
                diff_center_ys)
            idx_center_src = np.argmin(vertex_dist_from_center[1:]) + 1
            # center of mass
            com_calc_xs = abs(new_xs - np.average(new_xs))
            com_calc_ys = abs(new_ys - np.average(new_ys))
            coms = com_calc_xs + com_calc_ys
            idx_com_src = np.argmin(coms) if len(coms) > 0 else idx_center_src
            filtered_vertices = [[new_dict[idx] for idx in sublist] for sublist
                                 in filtered_vertices_1 if
                                 sublist[0] != idx_center_src and sublist[
                                     0] != idx_com_src] \
                                + [[new_dict[idx] for idx in sublist] for
                                   sublist in filtered_vertices_2] \
                                + [[new_dict[idx] for idx in sublist] for
                                   sublist in filtered_vertices_3]

            if not idx_center_src in new_vertices:
                filtered_vertices += [[idx_center_src]]
            if not idx_com_src in new_vertices and idx_com_src != idx_center_src:
                filtered_vertices += [[idx_com_src]]

        filtered_vertices = [tuple(sublist) for sublist in filtered_vertices]

        size = len(filtered_vertices)

        for _, row in self.src_df.iterrows():
            src_idx_enum = row[0]
            one_hot = row[1:]
            src_ids = tuple([i for i, b in enumerate(one_hot) if b])
            if src_ids not in filtered_vertices:
                filtered_df = filtered_df[
                    filtered_df['sourceIdx'] != src_idx_enum]

        print("amount source_combis: ", len(filtered_df))
        return filtered_df

    def read(self) -> [Graph]:
        if self.TIMEIT:
            print('start')
            last_time = datetime.now()

        for net_idx, row in self.input_df.iterrows():
            if self.start <= net_idx < self.end:
                graphDf = self.data_obj_df[
                    self.data_obj_df['netIdx'] == net_idx]
                net_idx_data = np.array(row, dtype=np.float32)
                self.dataset.append(MSPDMixedNet(
                    net_df=net_idx_data,
                    mode='train',
                    graph_df=graphDf,
                    filtered_src_df=self.filter_sources(net_idx_data[1:]),
                    **self.__dict__))

                if self.TIMEIT:
                    last_time = self.timeit(net_idx, last_time)
                else:
                    print(net_idx)

        if self.TIMEIT: print('end')
        return [g for net in self.dataset for g in net]

    @staticmethod
    def timeit(net_idx, last):
        print(net_idx)
        tmp = datetime.now()
        print((tmp - last).total_seconds())
        print('----')
        return tmp


def output_fn(num_classes):
    if num_classes == 1:
        return 'linear'
    elif num_classes == 2:
        return 'linear'  # 'sigmoid'
    else:
        return 'softmax'




class Mspd10(tf.keras.Model):
    def __init__(self, n_labels, dr_rate=0.4, n_hidden=512, **kwargs):
        super().__init__(kwargs)
        self.mask = GraphMasking()
        self.conv1 = GCNConv(32, mode='batch',
                             kernel_initializer=initializers.glorot_uniform())
        self.norm0 = layers.BatchNormalization()
        self.dropout0 = layers.Dropout(dr_rate)
        self.conv2 = GCNConv(32, mode='batch',
                             kernel_initializer=initializers.glorot_uniform(
                                 seed=0))
        self.norm01 = layers.BatchNormalization()
        self.dropout01 = layers.Dropout(dr_rate)
        self.conv3 = GCNConv(16, mode='batch',
                             kernel_initializer=initializers.glorot_uniform(
                                 seed=0))
        self.conv4 = GCNConv(8, mode='batch',
                             kernel_initializer=initializers.glorot_uniform(
                                 seed=0))
        self.norm1 = layers.BatchNormalization()
        self.dropout1 = layers.Dropout(dr_rate)
        self.flatten_sum2 = GlobalSumPool()
        self.flatten_max2 = GlobalMaxPool()
        self.flatten_avg2 = GlobalAvgPool()
        self.flat = layers.Flatten()
        self.pre_flat = layers.Flatten()
        self.dropout_pool = layers.Dropout(dr_rate)
        self.dense1 = layers.Dense(n_hidden, 'relu')

        self.norm2 = layers.BatchNormalization()
        self.dropout2 = layers.Dropout(dr_rate)
        self.dense2 = layers.Dense(n_hidden, 'relu')
        self.dense3 = layers.Dense(n_hidden // 2, 'relu')
        self.last = layers.Dense(n_labels, activation='linear')

    def call(self, inputs):
        x, a = inputs
        x = self.mask(x)
        x = self.conv1([x, a])
        x = self.dropout1(x)
        out_avg_pool2 = self.flatten_avg2(x)
        out_max_pool2 = self.flatten_max2(x)
        out = self.flat(tf.concat([out_avg_pool2, out_max_pool2], axis=1))
        out = self.dropout2(out)
        out = self.dense1(out)
        out = self.last(out)
        return out



ending = {
    10: 300,
    15: 1800,
    25: 1800,
    30: 1000,
    40: 500,
    45: 480,
    50: 400
}
ending_contest = {
    10: 300,
    15: 300,
    25: 300,
    30: 300,
    40: 300,
    45: 300,
    50: 300
}
split_val = 250
n = 10
d_train = MSPDTrainSet(N=n,start=0,end=split_val, **params)
d_valid = MSPDTrainSet(N=n,start=split_val,end= ending.get(n, 300), **params)
# contest = MSPDTrainSet(N=n,start=0,end= ending_contest.get(n, 300), **params)

train, valid = d_train, d_valid
np.random.shuffle(valid)
np.random.shuffle(train)

epochs = 400
batch_size = 128
patience = 25
data_early_stop = 0.00001
path_weights = f'{SAVE_PATH}/best_weights.h5'

classes = train.n_labels
out_fn = output_fn(classes)
is_regression = classes <= 2 or np.sum(train[0].y > train[0].n_nodes)

checkpoint_monitor = 'val_loss' if is_regression else 'val_acc'
mode = 'min' if is_regression else 'max'

early_stopping = callbacks.EarlyStopping(monitor=checkpoint_monitor,
                                         patience=patience,
                                         restore_best_weights=True,
                                         verbose=2,
                                         min_delta=data_early_stop,
                                         mode=mode)

checkpoint = callbacks.ModelCheckpoint(path_weights,
                                       monitor=checkpoint_monitor,
                                       verbose=1,
                                       save_freq='epoch',
                                       save_weights_only=True,
                                       save_best_only=True,
                                       mode=mode)

callbacks_list = [checkpoint, early_stopping]

shaper = PackedBatchLoader(train[:2000], batch_size=batch_size, mask=True)
batch = 0
for b in shaper:
    batch = b
    break

model = Mspd10(batch[1].shape[1], training=True)
model.training = True
model.compile(
    optimizer=tf.keras.optimizers.Adam(),
              loss= tf.keras.losses.MeanSquaredError(),
              metrics= ['acc'] if not is_regression else None)
model.training = True

build = model([batch[0][0], batch[0][1]])
model.training = True


loader_train = PackedBatchLoader(train, batch_size=batch_size,  epochs=epochs , mask=True)
loader_valid = PackedBatchLoader(valid ,batch_size=batch_size, mask=True)

history = model.fit(
    loader_train.load(),
    steps_per_epoch=loader_train.steps_per_epoch,
    validation_data=loader_valid.load(),
    validation_steps=loader_valid.steps_per_epoch,
    epochs=epochs,
    callbacks=callbacks_list,
    # class_weight=weights if not is_regression else None
)
# loader_test = PackedBatchLoader(contest,batch_size=batch_size, mask=True)
# print(model.evaluate(loader_test.load(),
#                      steps=loader_valid.steps_per_epoch,
#                      verbose=2))
# print(model.predict(loader_test.load(), steps=loader_valid.steps_per_epoch,verbose=2))