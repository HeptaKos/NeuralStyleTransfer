import tensorflow as tf
import numpy as np
import settings
import scipy.io
import scipy.misc
import tensorflow as tf
import imageio
import PIL.Image as Image


class Model(object):
    def __init__(self, content_path, style_path):
        self.content = self.loading(content_path)
        self.style = self.loading(style_path)
        self.random_img = self.get_random_img()
        self.net = self.vggnet()

    def vggnet(self):
        vgg = scipy.io.loadmat(settings.VGG_MODEL_PATH)
        vgg_layers = vgg['layers'][0]
        net = {}

        net['input'] = tf.Variable(np.zeros([1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3]), dtype=tf.float32)
        # 参数对应的层数可以参考vgg模型图
        net['conv1_1'] = self.conv_relu(net['input'], self.get_wb(vgg_layers, 0))
        net['conv1_2'] = self.conv_relu(net['conv1_1'], self.get_wb(vgg_layers, 2))
        net['pool1'] = self.pool(net['conv1_2'])
        net['conv2_1'] = self.conv_relu(net['pool1'], self.get_wb(vgg_layers, 5))
        net['conv2_2'] = self.conv_relu(net['conv2_1'], self.get_wb(vgg_layers, 7))
        net['pool2'] = self.pool(net['conv2_2'])
        net['conv3_1'] = self.conv_relu(net['pool2'], self.get_wb(vgg_layers, 10))
        net['conv3_2'] = self.conv_relu(net['conv3_1'], self.get_wb(vgg_layers, 12))
        net['conv3_3'] = self.conv_relu(net['conv3_2'], self.get_wb(vgg_layers, 14))
        net['conv3_4'] = self.conv_relu(net['conv3_3'], self.get_wb(vgg_layers, 16))
        net['pool3'] = self.pool(net['conv3_4'])
        net['conv4_1'] = self.conv_relu(net['pool3'], self.get_wb(vgg_layers, 19))
        net['conv4_2'] = self.conv_relu(net['conv4_1'], self.get_wb(vgg_layers, 21))
        net['conv4_3'] = self.conv_relu(net['conv4_2'], self.get_wb(vgg_layers, 23))
        net['conv4_4'] = self.conv_relu(net['conv4_3'], self.get_wb(vgg_layers, 25))
        net['pool4'] = self.pool(net['conv4_4'])
        net['conv5_1'] = self.conv_relu(net['pool4'], self.get_wb(vgg_layers, 28))
        net['conv5_2'] = self.conv_relu(net['conv5_1'], self.get_wb(vgg_layers, 30))
        net['conv5_3'] = self.conv_relu(net['conv5_2'], self.get_wb(vgg_layers, 32))
        net['conv5_4'] = self.conv_relu(net['conv5_3'], self.get_wb(vgg_layers, 34))
        net['pool5'] = self.pool(net['conv5_4'])
        return net

    def conv_relu(self,input, wb):
        conv = tf.nn.conv2d(input, wb[0], strides=[1, 1, 1, 1], padding='SAME')
        relu = tf.nn.relu(conv + wb[1])
        return relu

    def pool(self, input):
        """
        进行max_pool操作
        :param input: 输入层
        :return: 池化后的结果
        """
        return tf.nn.max_pool(input, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

    def get_wb(self, layers, i):
        """
        从预训练好的vgg模型中读取参数
        :param layers: 训练好的vgg模型
        :param i: vgg指定层数
        :return: 该层的w,b
        """
        w = tf.constant(layers[i][0][0][0][0][0])
        bias = layers[i][0][0][0][0][1]
        b = tf.constant(np.reshape(bias, (bias.size)))
        return w, b

    def get_random_img(self):
        """
        根据噪音和内容图片，生成一张随机图片
        :return:
        """
        noise_image = np.random.uniform(-20, 20, [1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3])
        random_img = noise_image * settings.NOISE + self.content * (1 - settings.NOISE)
        return random_img

    def loading(self, path):
        """
        加载一张图片，将其转化为符合要求的格式
        :param path:
        :return:
        """
        im = Image.open(path)
        out = im.resize((settings.IMAGE_WIDTH,settings.IMAGE_HEIGHT), Image.ANTIALIAS)  # resize image with high-quality
        out.save(path)
        image = imageio.imread(path)
        print(image)
        image = out.convert("RGB")
        image = np.array(image)
        image = np.reshape(image, (1, settings.IMAGE_HEIGHT, settings.IMAGE_WIDTH, 3))
        # 减去均值，使其数据分布接近0
        image = image - settings.IMAGE_MEAN_VALUE
        return image

if __name__ == '__main__':
    Model(settings.CONTENT_IMAGE, settings.STYLE_IMAGE)
