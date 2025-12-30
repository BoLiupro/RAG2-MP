# China_Journal Mobility Prediction
## 数据集
1. 原始数据(Raw)收集：
   1. mobility：南昌app（202205）、上海信令(2024)、深圳私家车(202207)
   2. POI：南昌(202203)、上海(2024)、深圳(202211)，GCJ-02
2. 数据集(Processed)处理:
   1. POI:
      + 确定研究空间范围
      + 网格划分，确定网格中心
      + 坐标系对齐、poi 分布统计+POI总数量
   2. Mobility
      + 现在有mobility数据和poi数据，mobility包括时间（到分秒）、位置（经纬度）、userid
      + 所有日期合并，按照user group
      + 按照user划分训练集、验证集、测试集  
      + 每个user按照obs_len+pred_len进行滑动窗口切片，得到samples
      + 所有samples进行检查，必须每一步都在研究空间范围内，位置由坐标系变换到对应Location_ID
      + 数据统计，调整参数

## 代码框架
