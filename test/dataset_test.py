import unittest
from trainer.datasets import oxford_iiit_pet_dataset, oxford_iiit_pet_dataset_D

class Test(unittest.TestCase):
  def test_oxford_iiit_pet_dataset(self):
    count = {
      'test': 3669,
      'train': 3680,
    }
    for split in count.keys():
      dataset, cnt = oxford_iiit_pet_dataset(split, size=(224,224,3), downsampling_factor=4, batch_size=32)
      n_iter = iter(dataset)
      data = next(n_iter)
      self.assertEqual(cnt, count[split])
      self.assertEqual(len(data), 2)
      self.assertEqual(data[0].shape, (32, 56, 56, 3))
      self.assertEqual(data[1].shape, (32, 224, 224, 3))

  def test_oxford_iiit_pet_dataset_D(self):
    count = {
      'test': 3669 * 2,
      'train': 3680 * 2,
    }
    for split in count.keys():
      dataset, cnt = oxford_iiit_pet_dataset_D(split, size=(224,224,3), downsampling_factor=4, batch_size=32)
      n_iter = iter(dataset)
      data = next(n_iter)
      self.assertEqual(cnt, count[split])
      self.assertEqual(len(data), 2)
      self.assertEqual(data[0].shape, (32, 224, 224, 3))
      self.assertListEqual(list(set(data[1].numpy())), [0, 1])


if __name__ == '__main__':
  unittest.main()
