# 车辆检测+计数+车牌检测与车牌识别
#### 介绍
基于pytorch深度学习框架，实用开源模型yolov4实现模板检测与yolov5实现车牌检测与LPRNet实现车牌检测

基于win10系统，实用anaconda配置python环境，在anaconda里面下载vscode对项目进行编辑，

#### 安装教程

```
conda create –n car-detect python=3.6
activate car-detect
pip install torch===1.2.0 torchvision===0.4.0 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```



#### 使用说明

1.  运行detect.py：实现对 /inference/images 路径下的图片和视频进行目标检测，卡车计数，和车牌检测与识别
2.  在/inference/output 路径下可看到输出情况