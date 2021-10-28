import os

# region pre-processing
name_dict = {"apple": 0, "banana": 1, "grape": 2, "orange": 3, "pear": 4}

data_root_path = "data/fruits/"
test_file_path = data_root_path + "test.txt"
train_file_path = data_root_path + "train.txt"
name_data_list = {}


def save_train_test_file(path, name):
    if name not in name_data_list:
        img_list = []
        img_list.append(path)
        name_data_list[name] = img_list
    else:
        name_data_list[name].append(path)


dirs = os.listdir(data_root_path)
for d in dirs:
    full_path = data_root_path + d
    if os.path.isdir(full_path):
        imgs = os.listdir(full_path)
        for img in imgs:
            save_train_test_file(full_path + "/" + img, d)

with open(test_file_path, "w"):
    pass

with open(train_file_path, "w"):
    pass

for name, img_list in name_data_list.items():
    i = 0
    num = len(img_list)

    for img in img_list:
        if i % 10 == 0:
            with open(test_file_path, "a") as f:
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        else:
            with open(train_file_path, "a") as f:
                line = "%s\t%d\n" % (img, name_dict[name])
                f.write(line)
        i += 1

print("pre-processing complete.")
# endregion

# region import_models
import paddle
import paddle.fluid as fluid
import numpy as np
import sys
import os
from multiprocessing import cpu_count
import matplotlib.pyplot as plt


# endregion

# region define_reader
def train_mapper(sample):
    img, lab = sample
    if not os.path.exists(img):
        print(f'{img} does not exist.')

    img = paddle.dataset.image.load_image(img)
    img = paddle.dataset.image.simple_transform(img, resize_size=100, crop_size=100, is_color=True, is_train=True)
    img = img.astype("float32") / 255.0
    return img, lab


def train_r(train_list, buffered_size=1024):
    def reader():
        with open(train_list, "r") as f:
            lines = [line.strip() for line in f]
            for line in lines:
                img_path, lab = line.replace("\n", "").split("\t")
                yield img_path, int(lab)

    return paddle.reader.xmap_readers(train_mapper, reader, cpu_count(), buffered_size)


# endregion

# region variants definition
BATCH_SIZE = 32

train_reader = train_r(train_list=train_file_path)

random_train_reader = paddle.reader.shuffle(reader=train_reader, buf_size=1300)
batch_train_reader = paddle.batch(random_train_reader, batch_size=BATCH_SIZE)
image = fluid.layers.data(name="image", shape=[3, 100, 100], dtype="float32")

label = fluid.layers.data(name="label", shape=[1], dtype="int64")


# endregion

# region define vgg
def vgg_bn_drop(image, type_size):
    def conv_block(ipt, num_filter, groups):
        return fluid.nets.img_conv_group(input=ipt,  # image data, [N, C, H, W]
                                         pool_stride=2,
                                         pool_size=2,
                                         conv_num_filter=[num_filter] * groups,
                                         conv_filter_size=3,
                                         conv_act='relu',
                                         conv_with_batchnorm=True,
                                         pool_type='max')

    conv1 = conv_block(image, 64, 2)
    conv2 = conv_block(conv1, 128, 2)
    conv3 = conv_block(conv2, 256, 3)
    conv4 = conv_block(conv3, 512, 3)
    conv5 = conv_block(conv4, 512, 3)

    drop = fluid.layers.dropout(x=conv5, dropout_prob=0.5)
    fc1 = fluid.layers.fc(drop, 512)
    bn = fluid.layers.batch_norm(fc1, act='relu')
    drop2 = fluid.layers.dropout(x=bn, dropout_prob=0.2)
    fc2 = fluid.layers.fc(drop2, 512)
    predict = fluid.layers.fc(fc2, type_size, act='softmax')
    return predict


# endregion

# region define optimizer and variants for training
predict = vgg_bn_drop(image=image, type_size=5)
cost = fluid.layers.cross_entropy(input=predict, label=label)
avg_cost = fluid.layers.mean(cost)
accuracy = fluid.layers.accuracy(input=predict, label=label)

optimizer = fluid.optimizer.Adam(learning_rate=0.0000001)

place = fluid.CPUPlace(0)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(feed_list=[image, label], place=place)

costs = []
accs = []
times = 0
batches = []
# endregion

# region training and model saving
epochs = 10

for pass_id in range(epochs):
    train_cost = 0
    for batch_id, data in enumerate(batch_train_reader()):
        times += 1
        train_cost, train_acc = exe.run(program=fluid.default_main_program(), feed=feeder.feed(data),
                                        fetch_list=[avg_cost, accuracy])
        if batch_id % 20 == 0:
            print("pass_id:%d, bat_id:%d, cost:%f, acc:%f" % (pass_id, batch_id, train_cost[0], train_acc[0]))

        accs.append(train_acc[0])
        costs.append(train_cost[0])
        batches.append(times)

model_save_dir = "model/fruits/"
if not os.path.exists(model_save_dir):
    os.makedirs(model_save_dir)

fluid.io.save_inference_model(dirname=model_save_dir, feeded_var_names=["image"], target_vars=[predict], executor=exe)

print('model saving complete.')
# endregion

# region performance visualization
plt.figure('training', facecolor='lightgray')
plt.title("training", fontsize=24)
plt.xlabel("iter", fontsize=20)
plt.ylabel("cost/acc", fontsize=20)
plt.plot(batches, costs, color='red', label="Training Cost")
plt.plot(batches, accs, color='green', label="Training Acc")
plt.legend()
plt.grid()
plt.show()
plt.savefig("train.png")
# endregion
