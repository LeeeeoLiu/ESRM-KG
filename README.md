# ESRM-KG

This implementation of ***Keywords Generation improves E-commerce Session-based Recommendation*** based on a fork of OpenNMT.

# Dataset 

## Description

Each click contains the session id, item id, and item title (separated by \t) :

> session_id**\t**item_sku_id**\t**item_title

Each session contains several clicks (separated by ||):

> Click**||**Click**||**Click




## Examples

Example of src file:
```
0	1167	vivo Z1 新 一代 全面 屏 AI 双 摄 手机 4GB + 64GB 炫慕 红移动 联通 电信 全 网通 4G 手机 双 卡 双待||0	1206	vivo Y83 刘海 全面 屏 4GB + 64GB 极 夜 黑 移动 联通 电信 4G 手机 双 卡 双待||0	1232	vivo Z1 新 一代 全面 屏 AI 双 摄 手机 4GB + 64GB 宝石 蓝 移动 联通 电信 全 网通 4G 手机 双 卡 双待||0	831	荣耀 畅 玩 7X 4GB + 32GB 全 网通 4G 全面 屏 手机 标配 版 铂 光金||0	1265	vivo Z1 新 一代 全面 屏 AI 双 摄 手机 4GB + 64GB 瓷釉 蓝移动 联通 电信 全 网通 4G 手机 双 卡 双待||0	1195	荣耀 9i 4GB + 64GB 魅 海蓝 移动 联通 电信 4G 全面 屏 手机 双 卡 双待
```

Example of tgt file:
```
0	1025	华为 HUAWEI nova 3e 全面 屏 2400万 前 置 摄像 4GB + 64GB 幻夜 黑全网 通版 移动 联通 电信 4G 手机 双 卡 双待
```



## Data Access

Signed the following copyright announcement with your name and organization. Then complete the form online(https://forms.gle/9tNuMgRQLpXyARqk8) and mail to yxliu#ir.hit.edu.cn ('#'->'@'), we will send you the corpus by e-mail when approved.



## Copyright

The original copyright of all the conversations belongs to the source owner.

The copyright of annotation belongs to our group, and they are free to the public.

The dataset is only for research purposes. Without permission, it may not be used for any commercial purposes and distributed to others.



# Code


```python
# python 3.6
pip install -r requirements.txt

# preprocess data
python3 preprocess.py

# train
python3 train.py

# translate
python3 translate.py -model path/to/saved/checkpoint/models
```

# Citation
We appreciate your citation if you find our dataset is beneficial.

```

```