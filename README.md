# webGS
A cloud and personal computer co-rendering methods 

## [Project page](https://github.com/DistilledW/webGS) | [Paper](https://github.com/DistilledW/webGS) 

This repository contains three parts including train, cloud and client. 

## Setup
### Prerequisite
### Python Evironment for optimization
## Running the method
### To delete 
在这个项目里：
- cloud & client & submodules是最终会留在project，剩下的都是方便测试而留下的 （在源代码的基础上添加了兴阳的flash tree traversal和祝贺的fast_hier以及刘峥的端云协同的部分）
- train 用于训练，实际为源代码 
- dataset说明：
    - skybox 如果是在local测试，可以不用这个文件夹，但是如果是远程测试则会需要，该数据大小为23MB 
    - ./dataset/generate_test.py文件用于生成所需要的测试文件 
    - viewpoints.txt文件，这里面有small city下的全部camera的数据 
    - sort_viewpoint用于排序viewpoints.txt文件 
- configs: for test 

更新说明： 
- 完善了传输机制，在带宽满足条件的基础上，可以不需要从本地读取数据的过程
- 实现了压缩算法 
- 加上了兴阳的flashTree Traversal
- 添加了测试代码 

关于测试：
- 我修改了测试的shell脚本：
    - bash run.sh /path/to/config.cfg 
    - config.cfg 在例如./client/configs文件夹下，其他类似

[Note] 
- 修改后还没来得及测试，可能有bug需要改，但是应该不多 （大概） 