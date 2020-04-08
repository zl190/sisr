from __future__ import absolute_import, division, print_function, unicode_literals
import tensorflow as tf
import numpy as np
from trainer.models.sisr import MySRResNet, Discriminator, MySRGAN


class ModelTest(tf.test.TestCase):
  def setUp(self):
    self.im_h = 224
    self.im_w = 224
    self.im_dims = 3
    self.factor = 4
    self.bs = 32
    
  def testSRResNet(self):
    model = MySRResNet(shape=(int(self.im_h/self.factor), int(self.im_w/self.factor), self.im_dims))()
    input = tf.ones([self.bs, int(self.im_h/self.factor), int(self.im_w/self.factor), self.im_dims])
    output = model(input)
    self.assertShapeEqual(np.zeros((self.bs, self.im_h, self.im_w, self.im_dims)), output)

  def testDiscriminator(self):
    model = Discriminator(shape=(self.im_h, self.im_w, self.im_dims))()
    input = tf.ones([self.bs, self.im_h, self.im_w, self.im_dims])
    output = model(input)
    self.assertShapeEqual(np.zeros((self.bs, 1)), output)

  def testSRGAN(self):
    model = MySRGAN(hr_shape=(self.im_h, self.im_w, self.im_dims),
                    lr_shape=(int(self.im_h/self.factor), int(self.im_w/self.factor), self.im_dims),
                    L1_LOSS_ALPHA = 100,
                    GAN_LOSS_ALPHA = 0.001,
                    NUM_ITER = 1)
    generator, discriminator = model.get_models()

    input = tf.ones([self.bs, int(self.im_h/self.factor), int(self.im_w/self.factor), self.im_dims])
    output = generator(input)
    self.assertShapeEqual(np.zeros((self.bs, self.im_h, self.im_w, self.im_dims)), output)
    input = tf.ones([self.bs, self.im_h, self.im_w, self.im_dims])
    output = discriminator(input)
    self.assertShapeEqual(np.zeros((self.bs, 1)), output)


if __name__ == '__main__':
  tf.test.main()
