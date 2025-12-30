# China_Journal Mobility Prediction
---
## 数据集
1. 原始数据(Raw)收集：
   1. mobility：南昌app（202205）、[上海信令(2024)](https://wangshangguang.github.io/telecom_dataset//)、深圳私家车(202207)
   2. POI：南昌(202203)、上海(2024)、深圳(202211)，但是目前都是GCJ-02坐标系，需要先对齐到WGS-84坐标系
2. 数据集(Processed)处理:
   1. POI数据处理:
      + 确定研究空间范围
        > 深圳：(114.02,22.51)~(114.18,22.67),(Δ0.16,Δ0.16）for total =>40*40=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
        >
        > 南昌：(115.78,28.58)~(115.94,28.74),(Δ0.16,Δ0.16）for total =>40*40=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
        >
        > 上海：(121.40,31.18)~(121.56,31.34),(Δ0.16,Δ0.16）for total =>40*40=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
      + 网格划分，确定网格中心
      + POI类型一共14类，分别为：【Transportation Facilities, Leisure & Entertainment, Companies & Enterprises, Healthcare, 
      Residential, Tourist Attractions, Automotive, Life Services, Science & Education & Culture, Shopping & Consumer Goods,
      Sports & Fitness, Hotels & Accommodations, Financial Institutions, Dining & Cusine】。中文分别对应:[交通设置、休闲娱乐、公司企业、医疗保健、
      商务住宅、旅游景点、汽车相关、生活服务、科教文化、购物消费、运动健身、酒店住宿、金融机构、餐饮美食]
      + 坐标系从GCJ-02转换成WGC-84、poi 分布统计+POI总数量
   2. Mobility数据处理：
      + 现在有mobility数据和poi数据，mobility包括时间（到分秒）、位置（经纬度）、userid
      + 所有日期合并，按照user group
      + 按照user划分训练集、验证集、测试集  
      + 每个user得到所有的轨迹，等时间间隔离散化
      + 每个user按照obs_len+pred_len进行滑动窗口切片，得到samples
      + 所有samples进行检查，必须每一步都在研究空间范围内，位置由坐标系变换到对应Location_ID
      + 数据统计，调整参数
      + 需要针对不同出行方式拟合
---
## 代码框架
