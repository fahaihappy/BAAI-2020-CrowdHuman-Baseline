本baseline基于Iterdet，略微修改了配置文件。该算法2020.5月才放出，是目前crowdhuman数据集的SOTA。

原工程地址 https://github.com/saic-vul/iterdet ，请参照源工程完成环境配置。

baseline得分0.7809.

### baseline运行步骤
- 下载数据集，放置在`data/crowd_human`目录
- 下载权重，放置在`work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/`目录下
- 运行转换程序
```
python tools/convert_datasets/crowd_human.py
```

- 运行测试程序
```python
python tools/test.py configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/crowd_human_full_faster_rcnn_r50_fpn_2x.pth --out result.pkl --eval bbox
```
运行结束后会在根目录产生`result.pkl`和`file_info.pkl`文件。

- 生成提交结果
由于CrowdHuman的结果格式为`(x,y,h,w)`，所以需要进行转换。
```
cd postprocessing
python postprocessing.py
```
将会在目录下生成`submission.txt`文件，提交即可。

- 线下测评
主办方在2019年的比赛中提供了测评代码，已经包含在本baseline中，在evaluate目录下。

若需要线下测评，可以按如下方式运行：
```
cd evaluate
python demo.py
```
实测线上成绩好于线下成绩，线下得分大约0.74。

### baseline算法说明
Iterdet算法本身可以参考论文，除了改方法，baseline中还进行了一些提分操作。

- add a Batch Normalization layer after each convolution layer to the FPN of both detectors, which slightly improves performance
- do not freeze the first block of ResNet as we add history together with the trainable convolution layer before this block



### 继续提分建议
强烈建议参考2019年比赛冠军的方案， https://zhuanlan.zhihu.com/p/68677880 ，他们讲的很详细了。

祝大家比赛顺利！
<!-- - image demo
```
python demo/image_demo.py /mnt/data/iterdet/data/crowd_human/Images/273271,c9db000d5146c15.jpg configs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x.py \
    work_dirs/iterdet/crowd_human_full_faster_rcnn_r50_fpn_2x/crowd_human_full_faster_rcnn_r50_fpn_2x.pth
``` -->