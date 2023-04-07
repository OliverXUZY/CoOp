# Datasets
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
tiered_imagenet path
```
tiered_imagenet/tiered_imagenet
|–– train/  # contains 351 folders like n01440764, n01443537, etc.
|–– val/    # contains 97 folders like n01440764, n01443537, etc.
|–– test/   # contains 160 folders like n01440764, n01443537, etc.
```

# Code
test CLIP zero shot without training
```
bash scripts/cocoop/zy_test.sh tiered_imagenet 1
```
