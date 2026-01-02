# China_Journal Mobility Prediction
---
# 数据集
1. 原始数据(Raw)收集：
   1. mobility：南昌app（202205）、[上海信令(2024)](https://wangshangguang.github.io/telecom_dataset//)、深圳私家车(202207)
   2. POI：南昌(202203)、上海(2014)、深圳(202211)，但是目前都是GCJ-02坐标系，需要先对齐到WGS-84坐标系
2. 数据集(Processed)处理:
   1. POI数据处理:
      + 确定研究空间范围
        > 深圳：(114.02,22.51)~(114.18,22.67),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
        >
        > 南昌：(115.78,28.58)~(115.94,28.74),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
        >
        > 上海：(121.40,31.18)~(121.56,31.34),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
        >
        > 北京：(116.31,39.84)~(116.47,40.00),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
      + 网格划分，确定网格中心
      + POI类型一共14类，分别为：【Transportation Facilities, Leisure & Entertainment, Companies & Enterprises, Healthcare, 
      Residential, Tourist Attractions, Automotive, Life Services, Science & Education & Culture, Shopping & Consumer Goods,
      Sports & Fitness, Hotels & Accommodations, Financial Institutions, Dining & Cusine】。中文分别对应:[交通设置、休闲娱乐、公司企业、医疗保健、
      商务住宅、旅游景点、汽车相关、生活服务、科教文化、购物消费、运动健身、酒店住宿、金融机构、餐饮美食]，对应POI.csv文件中大类。但是要注意的是,[生活服务]这个大类下面包含了一些过于频繁的中类，比如[公共厕所、公用电话]，这两个中类不考虑。
      + POI.csv坐标系从GCJ-02转换成WGS-84
      + 遍历每个小的网格，统计每个大类POI的数量，最终得到每个小网格的各种POI百分比和POI总数量。
   2. Mobility数据处理：
      + 现在有mobility数据和poi数据，mobility数据上基本包括时间（到分秒）、位置（经纬度）、userid。特别要注意的是nanchang的数据中location对应的经纬度坐标需要根据location_id再去location.csv中匹配获取经纬度坐标。
      + 所有日期合并，按照user进行group
      + 按照user划分训练集、验证集、测试集。就是一部分user用于测试，一部分user用于验证，一部分user用于训练  
      + 得到每个user所有的轨迹，等时间间隔离散化。时间间隔可以是半个小时、一个小时。这个可以是手动输入的参数。例如一个user轨迹原本是：2:05 in location_1, 2:33 in location_2, 2:45 in location_3, 3:10 in location_4。如果时间间隔是30分钟，那么离散化后就是：2:00 in location_1, 2:30 in location_2, 3:00 in location_4。中间缺失的位置可以用前一个位置进行填充，或者用特殊标记表示缺失位置;如果同一个时间间隔内有多个位置点，则取第一个位置点。
      + 离散化后，每个user按照obs_len+pred_len进行滑动窗口切片，得到samples。obs_len和pred_len可以作为参数输入。例如obs_len=6，pred_len=6，表示观察过去6个时间点，预测未来6个时间点的位置。默认obs_len=12,pred_len=1.
      + 所有samples进行检查，必须每一步都在研究空间范围内，位置由坐标系变换到对应Location_ID.符合的保留，不符合的丢弃。
      + 数据统计，调整参数
      + 需要针对不同出行方式拟合gravity model的分母
---
# Prompt_1.1[数据处理1]
## 背景
现在要对POI数据和用户的出行轨迹数据进行处理和分析，以便为后续的出行预测任务做准备。POI数据包含了不同类型的兴趣点信息，而出行轨迹数据记录了用户在不同时间点的位置变化。通过对这些数据的处理，可以提取出有用的特征，帮助我们更好地理解用户的出行行为模式。
研究空间范围均为正方形空间，具体左下和右上顶点如下：
> 深圳：(114.02,22.51)~(114.18,22.67),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
>
> 南昌：(115.78,28.58)~(115.94,28.74),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
>
> 上海：(121.40,31.18)~(121.56,31.34),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid
POI一共14类，分别为:[交通设置、休闲娱乐、公司企业、医疗保健、商务住宅、旅游景点、汽车相关、生活服务、科教文化、购物消费、运动健身、酒店住宿、金融机构、餐饮美食]，对应POI.csv文件中[大类]。但是要注意的是,[生活服务]这个大类下面包含了一些过于频繁的中类，比如[公共厕所、公用电话]，这两个中类不考虑。

## 任务
请完成以下任务：
1. **POI数据处理**：
   - 确定研究空间范围，并将其划分为40x40的小网格。
   - POI数据的坐标系从GCJ-02转换为WGS-84。
   - 遍历每个小网格，将POI数据按照预定义的网格划分进行统计，计算每个网格内各类POI的百分比和总POI的数量。
   - 将结果保存为csv文件

## 约束
- 使用Python编程语言。
- 使用pandas和numpy等常用数据处理库。

## 输出格式
- 编写一个Python脚本，完成上述任务，并生成处理后的POI数据csv文件。
- 一些关键的参数，比如研究空间范围、网格划分个数等，可以作为脚本的输入参数进行配置。
- 有适当的注释和过程打印

---
# Prompt_1.2[数据处理2]   
## 背景
现在要对用户的出行轨迹数据进行处理和分析，以便为后续的出行预测任务做准备。出行轨迹数据记录了用户在不同时间点的位置变化。通过对这些数据的处理，可以提取出有用的特征，帮助我们更好地理解用户的出行行为模式。我已经对研究空间范围进行了网格划分并编号，每个网格的poi信息也已经统计好了。现在有三个城市的用户出行轨迹数据，分别是南昌、上海和深圳。mobility数据上基本包括时间（到分秒）、位置（经纬度）、userid，已经基于WGS-84。特别要注意的是nanchang的数据中location对应的经纬度坐标需要根据location_id再去location.csv中匹配获取经纬度坐标。

## 任务
请完成以下任务：
1. **出行轨迹数据处理**：
   - 对于每个数据集，将所有日期合并，按照user进行group。
   - 按照user划分训练集、验证集、测试集。就是一部分user用于测试，一部分user用于验证，一部分user用于训练。
   - 得到每个user所有的轨迹，等时间间隔离散化。时间间隔可以是半个小时、一个小时。这个可以是手动输入的参数。例如一个user轨迹原本是：2:05 in location_1, 2:33 in location_2, 2:45 in location_3, 3:10 in location_4。如果时间间隔是30分钟，那么离散化后就是：2:00 in location_1, 2:30 in location_2, 3:00 in location_4。中间缺失的位置可以用前一个位置进行填充，或者用特殊标记表示缺失位置;如果同一个时间间隔内有多个位置点，则取第一个位置点。locatoin_id一定要和poi得到的结果严格对应，不要出现不匹配的情况。
   - 离散化后，每个user按照obs_len+pred_len进行滑动窗口切片，得到samples。obs_len和pred_len可以作为参数。输入。例如obs_len=6，pred_len=6，表示观察过去6个时间点，预测未来6个时间点的位置。默认obs_len=12,pred_len=1.
   - 所有samples进行检查，必须每一步都在研究空间范围内，位置由坐标系变换到对应Location_ID.符合的保留，不符合的丢弃。
   - 数据统计

## 约束
- 使用Python编程语言。
- 使用pandas和numpy等常用数据处理库。

## 输出格式
- 编写一个Python脚本，完成上述任务，并生成处理后的出行轨迹数据csv文件。
- 一些关键的参数，比如时间间隔、obs_len、pred_len等，可以作为脚本的输入参数进行配置。
- 有适当的注释和过程打印

## 输出示例
- 例如得到南昌市处理后出行轨迹数据csv文件，包括train.csv, val.csv, test.csv
- 每个csv文件包含user_id, timestamp, location_id等字段.user_id和raw_data中的可以对应上，location_id和poi处理后的结果可以对应上，timestamp是离散化后的时间戳,yymmdd hh:mm格式
---
# Prompt_1.3[数据处理修改]
## 背景
现在已经编写好了poi_processing.py和mobility_processing.py两个脚本，分别用于处理POI数据和用户出行轨迹数据。但是我现在更换了数据集，把raw_data下的shanghai数据集更换成了Beijing数据集，想要在新的数据集上运行这两个脚本。新的数据集的格式和之前的数据集略有不同，所以需要对脚本进行一些修改，以适应新的数据格式。北京的研究范围是：北京：(116.31,39.84)~(116.47,40.00),(Δ0.16,Δ0.16）for total =>划分为40*40小网格=>(Δ0.004,Δ0.004)≈(Δ400m,Δ400m) for each grid

## 任务
请完成以下任务：
1. **脚本修改**：
   - 修改poi_processing.py脚本，使其能够处理新的Beijing POI数据格式。
   - 修改mobility_processing.py脚本，使其能够处理新的Beijing出行轨迹数据格式。
   - 确保修改后的脚本能够正确读取新的数据格式，并完成之前定义的POI数据处理和出行轨迹数据处理任务。

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分,原来关于上海数据集的处理代码需要替换为北京数据集的处理代码，南昌和深圳的数据处理代码保持不变。
- 有一个细节需要修改，在mobility_processing.py脚本中，最终输出的time格式需要从yymmdd hh:mm格式修改为yyyymmdd hh:mm格式。
- 过程中需要有适当的注释和过程打印。

## 输出格式
- 提交修改后的poi_processing.py和mobility_processing.py脚本。
- 重新运行修改后的脚本，生成处理后的所有城市出行轨迹数据csv文件，包括train.csv, val.csv, test.csv。
---
# Prompt_1.4[数据处理：训练集验证集测试集划分]
## 背景
现在已经基本成功地运行完了所有的数据处理流程，但是我觉得现在的训练集、验证集和测试集的划分比例不是很合理。我想要把这个比例调整为80%训练集，10%验证集，10%测试集。这样可以让模型有更多的数据进行训练，从而提升模型的性能。

## 任务
请完成以下任务：
1. **数据集划分调整**：
   - 修改mobility_processing.py脚本中的数据集划分部分，将训练集、验证集和测试集的比例调整为80%、10%和10%。
   - 确保修改后的脚本能够正确地按照新的比例划分数据集，并生成相应的train.csv, val.csv和test.csv文件。
   - 最后还有能够打印处理完之后数据集的信息，比如用户数量、数据数量、范围分布、时间分布、数据集划分比例

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。   

## 输出格式
- 提交修改后的mobility_processing.py脚本。
- 重新运行修改后的脚本，生成处理后的所有城市出行轨迹数据csv文件，包括train.csv, val.csv, test.csv。文件全部保存在data/{city}目录下。
---
# 代码框架
---
我这个项目解决的问题是human mobility prediction,即根据user历史轨迹来预测下一步出现的位置。数学公式定义是：input包括user的历史轨迹S={s_1,s_2,...,s_i},每个s_i=(li,ti)，表示在时间ti出现在grid li (li是spatial identifier)位置上。output是下一个时间点t_(i+1)出现的位置l_(i+1)。我的method主要是将LLM和改进版的garvity model结合起来进行预测。主要包含三个module：
1. LLM+RAG:利用LLM对user的历史轨迹进行编码，得到user出行模式和意图的embedding表示，再将embedding与经验池(预先整理好的sample embedding,都是对obs_len+pred_len的sample进行编码得到的)中的数据中进行embedding similarity计算，找到top-m相似的历史轨迹。将这些similar samples的历史轨迹和next location整合成一个prompt输入到LLM中，得到一个“同类型轨迹可能下一个目的地去哪里”的总结概括similary_mob_summary，用于预测辅助。这里m是一个手动输入的参数，表示选择多少个相似样本。
2. 利用改进版的gravity model为next location的预测选择提供一些candidates。在当前位置（也就是在obs_len的最后一个时间步时user处于的位置），遍历一定范围内的grids，依据grids中的各个种类的poi数量信息，用改进版的gravity model公式:score_of_poi_A=weight*(num of category A in origin grid)/distance^2，计算每个grid中各个类型poi的score。最终得到每个poi类型下score最高的top-n grids作为candidates。这里有三个手动输入的参数，一个是引力公式的参数weight,另一个是candidates的数量n；第三个是遍历的范围radius。
3. 综合LLM+RAG module的输出similar summary和gravity model module的输出candidate grids，进行最终的next location预测.设计一个prompt，将similar summary和candidate grids的信息整合进去，输入到LLM中，得到最终的预测结果。最终的结果是一个top-K个预测位置列表。
4. 评估指标：top-K accuracy, MRR等。
5. 消融实验：变体一：不用LLM+RAG的信息(w/o RAG)；变体二：不用gravity model输出的信息(w/o Gravity)；变体三：最终不用LLM进行预测（w/o LLM Predictor).
6. 前期准备：6.1：RAG module的经验池准备：对训练集中的所有samples进行LLM编码，得到embedding，存储下来，作为经验池；6.2:拟合gravity model的参数，选择最优的weight参数和radius参数作为实验的默认值。
---
# Prompt_2.1[方法实现:LLM+RAG模块]
## 背景
在出行预测任务中，利用用户的历史轨迹数据来预测其未来的位置是一个关键问题。为了提升预测的准确性，我们计划结合大语言模型（LLM）和检索增强生成（RAG）技术。通过对用户历史轨迹进行编码，并与预先整理好的经验池进行相似性计算，我们可以为预测任务提供有价值的辅助信息。

## 任务
请完成以下任务：
1. **LLM+RAG模块实现**：
   - 编写一个Python函数，接受用户的历史轨迹数据作为输入。