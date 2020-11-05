# 【Windows平台】Tensorflow + Keras 安装



## Tensorflow+keras安装

- Tensorflow版本问题：

  有些很老的cpu无AVX2指令集，需要下载对应版本：[https://github.com/fo40225/tensorflow-windows-wheel](https://github.com/fo40225/tensorflow-windows-wheel)

  **注意对应python版本**，即是文件名中有cpxx的，xx对应37、36等。

- Keras和Tensorflow版本对应：[https://docs.floydhub.com/guides/environments/](https://docs.floydhub.com/guides/environments/)

实测：

tensorflow 1.12 + keras 2.2.4 + python 3.7





## 使用Bert

 使用`bert-tensorflow`(pip install bert-tensorflow)，在tensorflow2.0.0和tensorflow1.12下都会报错，不知道问题出现在哪，还是老老实实把bert放在了项目下。

上述环境可行，bert的依赖项是`tensorflow > 1.11.0`



