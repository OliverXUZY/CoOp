# Multitask finetuning using CoCoOp

## Result
In finetuning stage, sample 5 classes per batch, 1 image per class, perform CoCoOp tuning the simple neural net generating image specific token.
Context: `'a photo of a {}',`

#### 160-way accuracy (%) on *tiered-ImageNet*
| pre-train  | zero-shot NC     | Multi-task finetune + NC |
|------------|------------------|--------------------------|
|CLIP-ViT_B32| 69.9   |  71.4           | 

## Datasets
```
$DATA = /srv/home/zxu444/datasets/
$DATA/
|–– imagenet/
|–– tiered-imagenet/tiered_imagenet
```
imagenet path
```
imagenet/
|–– images/
|   |–– train/ # contains 1,000 folders like n01440764, n01443537, etc.
|   |–– val/
```
tiered_imagenet path (Downloaded cached files [here](https://drive.google.com/file/d/1PSpCTF6U6bzOqWp0jF4XhhhybIpc3di8/view?usp=sharing)
)
```
tiered_imagenet/tiered_imagenet
|–– train/  # contains 351 folders like n01440764, n01443537, etc.
|–– val/    # contains 97 folders like n01440764, n01443537, etc.
|–– test/   # contains 160 folders like n01440764, n01443537, etc.
|–– cached_test_labels_vl-tiered-imagenet.npy
|–– cached_train_labels_vl-tiered-imagenet.npy
|–– cached_val_labels_vl-tiered-imagenet.npy
```
`cached_val_labels_vl-tiered-imagenet.npy` is a long vector represents labels: `array([ 0,  0,  0, ..., 96, 96, 96])` (97 classes in val dataset). 

Inside `tiered_imagenet.py`, `self.catlocs` for val data will look like (around 1300 images per class):
```
(array([   0,    1,    2, ..., 1297, 1298, 1299]),
 array([1300, 1301, 1302, ..., 2515, 2516, 2517]),
 array([2518, 2519, 2520, ..., 3815, 3816, 3817]),
 array([3818, 3819, 3820, ..., 5115, 5116, 5117]),
 array([5118, 5119, 5120, ..., 6415, 6416, 6417]),
 array([6418, 6419, 6420, ..., 7715, 7716, 7717]),
 array([7718, 7719, 7720, ..., 9015, 9016, 9017]),
 .....
 array([122961, 122962, 122963, ..., 124258, 124259, 124260]))
```


## Code
test CLIP zero shot without training
```
bash scripts/cocoop/zy_test.sh tiered_imagenet 1
```
This will get me 0.1%.
