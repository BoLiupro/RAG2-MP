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
# Prompt_1.1[POI数据处理]
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
# Prompt_1.2[Mobility数据处理]   
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
# Prompt_1.5[数据集划分调整]
## 背景
现在已经写好了可以用的mobility_process.py和poi_process.py脚本，分别用于处理用户出行轨迹数据和POI数据。但是我想做三点调整：
1、现在是按照每个user进行train/val/test划分的。但是我想还划分一部分user用于建立RAG经验池。也就是说，现在的数据集划分应该是rag/train/val/test四个部分。rag的数据用于后续的RAG模块，作为经验池的数据来源。比例按照0.4,0.4,0.1,0.1划分。这个比例可以作为一个输入参数进行控制。
2、划分好并生成完mobility数据之后，目前的数据量有点太多了。我想对每个数据集进行随机采样，每个部分都随机取一定百分比的sample。注意是对生成好了的sample采样，而不是对user进行采样，我需要保证数据尽可能多地覆盖不同的user。这个百分比可以作为一个输入参数进行控制。
3、现在是按照半个小时的时间间隔进行离散化的。我需要一个输入参数的位置来控制时间间隔，可以选择15分钟、30分钟、60分钟等不同的时间间隔进行离散化。单位是hours。
## 任务
请完成以下任务：
1. **数据集划分和采样调整**：
   - 修改mobility_processing.py脚本中的数据集划分部分，将数据集划分为rag/train/val/test四个部分，比例按照0.4,0.4,0.1,0.1划分。这个比例作为输入参数进行控制。
   - 在生成完每个数据集之后，添加一个随机采样的步骤，对每个数据集进行随机采样，采样比例作为输入参数进行控制。
   - 修改时间间隔离散化的部分，添加一个输入参数来控制时间间隔，单位为hours。
## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。
## 输出格式
- 提交修改后的mobility_processing.py脚本。
- 重新运行修改后的脚本，生成处理后的所有城市出行轨迹数据csv文件，包括rag.csv, train.csv, val.csv和test.csv。文件全部保存在data/{city}目录下。

---
# 代码框架
---
我这个项目解决的问题是human mobility prediction,即根据user历史轨迹来预测下一步出现的位置。数学公式定义是：input包括user的历史轨迹S={s_1,s_2,...,s_i},每个s_i=(li,ti)，表示在时间ti出现在grid li (li是spatial identifier)位置上。output是下一个时间点t_(i+1)出现的位置l_(i+1)。我的method主要是将LLM和改进版的garvity model结合起来进行预测。主要包含三个module：
1. LLM+RAG:利用LLM对user的历史轨迹进行编码，得到user出行模式和意图的embedding表示，再将embedding与经验池(预先整理好的sample embedding,都是对obs_len+pred_len的sample进行编码得到的)中的数据中进行embedding similarity计算，找到top-m相似的历史轨迹。将这些similar samples的历史轨迹和next location整合成一个prompt输入到LLM中，得到一个“同类型轨迹可能下一个目的地去哪里”的总结概括similary_mob_summary，用于预测辅助。这里m是一个手动输入的参数，表示选择多少个相似样本。
2. 利用改进版的gravity model为next location的预测选择提供一些candidates。在当前位置（也就是在obs_len的最后一个时间步时user处于的位置），遍历一定范围内的grids，依据grids中的各个种类的poi数量信息，用改进版的gravity model公式:score_of_poi_A=weight*(num of category A in origin grid)/distance^2，计算每个grid中各个类型poi的score。最终得到每个poi类型下score最高的top-n grids作为candidates。这里有三个手动输入的参数，一个是引力公式的参数weight,另一个是candidates的数量n；第三个是遍历的范围radius。
3. 综合LLM+RAG module的输出similar summary和gravity model module的输出candidate grids，进行最终的next location预测.设计一个prompt，将similar summary和candidate grids的信息整合进去，输入到LLM中，得到最终的预测结果。最终的结果是一个top-K个预测位置列表。为了减小模型参数量，RAG使用的LLM和最终预测使用的LLM是同一个模型LLM，不加载两个LLM。
---
# Prompt_2.1[框架搭建：LLM+RAG模块实现]
## 背景
现在我要搭建LLM+RAG模块的代码框架。这个模块的主要功能是利用大语言模型(LLM)对用户的历史轨迹进行编码，生成embedding表示。然后将这些embedding与预先整理好的经验池中的数据进行相似度计算，找到top-m相似的历史轨迹。最后将这些相似样本的历史轨迹和下一个位置整合成一个prompt，输入到LLM中，得到一个“同类型轨迹可能下一个目的地去哪里”的总结概括，用于预测辅助。

## 任务
请完成以下任务：
1. **LLM+RAG模块实现**：
   - 编写一个Python函数，接受用户的历史轨迹数据（obs）作为输入。
   - 加载本地的大预言模型，位置在/datadisk/{LLM Backbone},例如Deepseek-R1-Distill-Qwen-3B.
   - 基于历史轨迹上下文（包括location地点、poi信息、时间），构建一个合适的prompt（beijing和nanchang数据集是手机信令和gps，没有侧重的出行方式，但shenzhen都是私家车的轨迹，表明是基于私家车出行的移动方式），利用LLM对用户的历史轨迹进行编码，生成embedding表示。
   - 加载预先整理好的经验池数据，这些数据已经经过LLM编码，存储在指定路径:/workspace/China_Journal/model/rag_database。
   - 计算输入历史轨迹的embedding与经验池中所有样本的embedding之间的余弦相似度，找到top-m相似的历史轨迹样本。
   - 将这些相似样本的历史轨迹和下一个位置整合成一个prompt，输入到LLM中，得到一个“同类型轨迹可能下一个目的地去哪里”的总结概括，用于预测辅助。

## 约束
- 使用Python编程语言。
- 使用from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig等库加载、量化和使用LLM。
- 使用from peft import get_peft_model, LoraConfig, TaskType微调模型。
- 先专注于框架的搭建，训练流程、损失函数、RAG库的建立等部分后续再完善。临时性的测试流程可以先写在/workspace/China_Journal/trainer/trainer.py
- 输出的summary是text。

## 输出格式
- 提交两个Python脚本文件，一个是RAG.py用于走rag的流程，一个LLm.py用于初始化LLM和利用LLM进行推理。这个LLM.py涉及RAG和最终预测模块，不是两个独立的LLM。
- 所有的类用大写开头驼峰命名法，函数和变量用小写字母加下划线命名法。
- 提交的LLM.py脚本中，包括建立加载LLM、编码轨迹得到embedding的函数、接收similar samples并生成summary输出的函数。输入参数包括LLM名（例如Deepseek-R1-Distill-Qwen-3B）、LLM维度相关参数等。
- 提交RAG.py脚本中，包括接收obs，调用LLM编码得到embedding，计算相似度，找到top-m相似样本，调用LLM生成summary的函数。输入参数包括经验池路径、top-m参数等。
---
# Prompt_2.2[框架搭建：改进版GravityModel模块实现]
## 背景
现在我要搭建改进版Gravity Model模块的代码框架。这个模块的主要功能是利用改进版的重力模型为下一个位置的预测选择一些候选位置。在当前位置，遍历一定范围内的网格，依据网格中的各个种类的POI数量信息，计算每个网格中各个类型POI的得分。最终得到每个POI类型下得分最高的top-n网格作为候选位置。

## 任务
请完成以下任务：
1. **改进版Gravity Model模块实现**：
   - 编写一个Python函数，接受用户的当前位置信息（当前位置所在的网格ID）和POI数据作为输入。
   - 遍历当前位置一定范围内的网格，依据网格中的各个种类的POI数量信息，计算每个网格中各个类型POI的得分，使用改进版的重力模型公式:score_of_poi_A=weight*(num of category A in origin grid)/distance^2。
   - 最终得到每个POI类型下得分最高的top-n网格作为候选位置。

## 约束
- 使用Python编程语言。
- 使用pandas和numpy等常用数据处理库。
- 先专注于框架的搭建，训练流程、损失函数等部分后续再完善。临时性的测试流程可以先写在/workspace/China_Journal/trainer/trainer.py
- 输入参数包括weight参数、候选位置数量n、遍历范围radius等。

## 输出格式
- 提交一个Python脚本文件，命名为Gravity.py。
- 所有的类用大写开头驼峰命名法，函数和变量用小写字母加下划线命名法。
- 提交的Gravity.py脚本中，包括接收当前位置和POI数据，计算各个网格POI得分，选择top-n候选位置的函数。输入参数包括weight参数、候选位置数量n、遍历范围radius等。
---
# Prompt_2.3[当前RAG+Gravity功能检查与修改]
## 背景
现在写的LLM+RAG和Gravity模块基本上正确且可以运行了。但是有一些地方我觉得不太对，需要修改。现在有以下几个问题：
1、Gravity模块最后的输出，现在是把所有poi类别合并在一起，选出了top-n个候选位置。但是我想要的是针对每个poi类别，分别选出top-n个候选位置。也就是说，如果有14个poi类别，最终的输出应该是14组top-n候选位置，每组对应一个poi类别。这样LLM在结合similar summary分析了user的mobility的意图之后，可以针对不同的poi类别，选择不同的候选位置进行预测。
2、LLM+RAG模块中，embedding相似度计算的部分，现在是直接计算输入轨迹embedding和经验池中所有样本的embedding之间的余弦相似度。这样计算量比较大，效率不高。我想要改成使用faiss库来进行相似度计算。faiss是一个高效的相似度搜索库，可以大大提升相似度计算的效率。需要把经验池中的embedding存储成faiss索引，然后使用faiss进行相似度搜索，找到top-m相似样本。
3.现在在测试的过程中，我想吧输入给LLM的prompt打印出来，看看具体是什么内容。这样可以帮助我调试和优化prompt设计。

## 任务
请完成以下任务：
1. **Gravity模块修改**：
   - 修改Gravity.py脚本中的输出部分，使其针对每个POI类别，分别选出top-n个候选位置。
2. **LLM+RAG模块修改**：
   - 修改RAG.py脚本中的相似度计算部分，使用faiss库进行相似度搜索，找到top-m相似样本。
   - 在RAG.py脚本中，添加打印输入给LLM的prompt的功能。


---
# Prompt_2.4[框架搭建：最终预测模块实现]
## 背景
我想写一个py类，来实现整个流程的控制。这个类的主要功能是综合利用LLM+RAG模块和改进版Gravity Model模块，并最终使用LLM进行最终的下一个位置预测。具体来说，这个类需要完成以下几个步骤：
1. 接收用户的历史轨迹数据（obs）作为输入。
2. 调用LLM+RAG模块，获取“同类型轨迹可能下一个目的地去哪里”的总结概括summary。
3. 调用改进版Gravity Model模块，获取每个POI类别下的top-n候选位置candidates。
4. 设计一个合适的prompt，将summary和candidates的信息整合进去，输入到LLM中，得到最终的预测结果。

## 任务
请完成以下任务：
1. **最终预测模块实现**：
   - 编写一个Python类，命名为MobilityPredictor。
   - 在类的初始化方法中，加载LLM、RAG模块和改进版Gravity Model模块。
   - 编写一个方法，接受用户的历史轨迹数据（obs）作为输入，调用LLM+RAG模块获取summary，调用Gravity Model模块获取candidates。
   - 设计一个合适的prompt，将summary和candidates的信息整合进去，输入到LLM中（原先已经加载，只不过现在任务不同），得到最终的预测结果。

## 约束
- 使用Python编程语言。
- 使用之前编写的LLM.py, RAG.py和Gravity.py脚本中的类和函数。
- 先专注于框架的搭建，训练流程、损失函数等部分后续再完善。临时性的测试流程可以先写在/workspace/China_Journal/trainer/trainer.py
- 输入参数包括obs数据、top-K预测位置、RAG中的similar sample的数量、gravity的radius等所有模块中我之前提到过需要手动设置的参数，都要先输入给MobilityPredictor类,再由MobilityPredictor类传递给各个子模块。

## 输出格式
- 提交一个Python脚本文件，命名为MobilityPredictor.py。
- 所有的类用大写开头驼峰命名法，函数和变量用小写字母加下划线命名法。
- 提交的MobilityPredictor.py脚本中，包括MobilityPredictor类的定义和实现。输入参数包括obs数据、top-K预测位置、RAG中的similar sample的数量、gravity的radius等所有模块中我之前提到过需要手动设置的参数。
- 最终利用LLM进行预测的prompt和回答也要打印出来，方便调试和优化prompt设计。
---
# Prompt_2.5[框架优化与调整]
## 背景
现在我已经基本完成了整个mobility prediction的代码框架，包括LLM+RAG模块、改进版Gravity Model模块和最终预测模块。但是在实际测试过程中，我发现有一些地方需要优化和调整：
1、LLM的回答有时候会被截断，导致输出不完整。我想要调整LLM的生成参数，比如增加max_new_tokens的值，确保回答的完整性。
2、RAG_Summary我觉得信息很混乱，不够结构化。比如现在的summary是一个长段落的文本，我觉得可以把summary设计成一个更结构化的格式，比如每个可能的下一个位置对应一个简短的描述，或者用列表的形式列出几个可能的位置和对应的理由。这样可以帮助LLM更好地理解和利用这些信息进行最终预测。
3、我觉得在similar sample输入到LLM、gravity candidates输入到LLM和最终输入到LLM的信息还需要加上空间的信息。我觉得可以加上location之间的距离，比如轨迹序列中location之间的距离、similar sample的轨迹中每个location之间的距离、gravity candidates与当前location的距离等。这样可以帮助LLM更好地理解位置之间的空间关系，从而提升预测的准确性。
4、gravity model输出的candidates还需要包含原地自身作为一个候选位置。有时候用户可能会选择留在原地不动，这种情况也需要考虑进去。
5、top-k,top-m、top-n等参数的命名需要更清楚一些。我觉得可以把top-k改成top_k_predictions，把top-m改成rag_top_m_samples，把top-n改成gravity_top_n_candidates。这样命名更清晰，能够更好地表达这些参数的含义。
6、输入给LLM的prompt我感觉让LLM以为这是一个不断在移动的轨迹，导致LLM太过注重移动速度。但是事实上mobility并不是一个”物理题目“，轨迹可能是移动和停止的结合体。我觉得需要在prompt中明确指出这一点，告诉LLM轨迹中可能包含停留不动的情况。
7、一些重复使用的函数工具，可以统一移动到utils.py中，方便调用和维护。
8、我想把rag_database挪到/workspace/China_Journal/util路径下。

## 任务
请完成以下任务：
1. **LLM参数调整**：
   - 修改LLM.py脚本中的生成参数，增加max_new_tokens的值，确保回答的完整性。
2. **RAG Summary格式优化**：
   - 修改RAG.py脚本中的summary生成部分，设计一个更结构化的summary格式。
3. **空间信息添加**：
   - 修改RAG.py和Gravity.py脚本，添加位置之间的距离信息。
4. **Gravity Candidates优化**：
   - 修改Gravity.py脚本，确保输出的candidates中包含原地自身位置。
5. **参数命名调整**：
   - 修改所有相关脚本中的top-k, top-m, top-n参数命名，分别改为top_k_predictions, rag_top_m_samples, gravity_top_n_candidates。
6. **Prompt优化**：
   - 修改RAG.py和MobilityPredictor.py脚本中的prompt设计，明确指出轨迹中可能包含停留不动的情况。
7. **工具函数整理**：
   - 将重复使用的函数工具统一移动到utils.py中。
8. **RAG数据库路径修改**：
   - 修改RAG.py脚本中的rag_database路径，改为/workspace/China_Journal/util/rag_database。
---
# Prompt_2.6[框架代码细节修改]
## 背景
现在我已经完成了整个mobility prediction的代码框架，并且进行了优化和调整。但是在实际测试过程中，我发现有一些代码细节需要修改和完善：
1、在距离的计算当中，我希望使用haversine公式来计算两个经纬度坐标之间的距离。这样可以更准确地反映地球表面的距离关系，而不是用多少个grid来表示距离，因为grid的大小是人为设定的，不能准确反映实际距离，LLM在理解空间关系时可能会有疑问。
2、现在rag_summary好像仅仅是把simlar samples的内容简单拼接在一起，我觉得可以用LLM对这些similar samples进行一些归纳总结，提取出更有代表性的特征和模式，而不是简单地拼接。这样可以让summary更加精炼和有用。
3、描述current trajectiry的时候，用”→ distance to next: 1.0 grids“我觉得有点奇怪，改成”distance to previous:“可能会好一些。因为LLM是根据历史轨迹来预测下一个位置的，描述距离时更关注前一个位置和当前位置之间的距离关系。
4、我移动了util.py的位置，现在放在/workspace/China_Journal/common/utils.py下，所以需要修改所有引用util.py的脚本中的import路径。

## 任务
请完成以下任务：
1. **距离计算修改**：
   - 修改utils.py脚本中的距离计算函数，使用haversine公式来计算两个经纬度坐标之间的距离。
2. **RAG Summary生成修改**：
   - 修改RAG.py脚本中的summary生成部分，使用LLM对similar samples进行归纳总结，提取出更有代表性的特征和模式。
3. **轨迹距离描述修改**：
   - 修改RAG.py脚本中的轨迹描述部分，将”distance to next“改为”distance to previous“。
4. **util.py路径修改**：
   - 修改所有引用util.py的脚本中的import路径，改为from common.utils import ...

## 约束
1、RAG_summary的生成部分需要调用LLM进行归纳总结，而不是简单拼接similar samples。prompt要引导从similar samples中提取出代表性的特征和模式并提供一个回答的模板，避免LLM
的回答过于发散和格式过于混乱。
2、RAG_suammary的长度可能需要控制，避免出现过长summry然后被截断的情况。我觉得可以设置一个最大长度的限制。
---


# 训练流程 
---
我现在需要设计整个模型的训练流程。主要包括以下几个部分：
1. 数据加载：加载处理好的训练集、验证集和测试集数据。
2. 模型初始化：初始化LLM、RAG模块和Gravity模块。主要是调用predictor.py中的MobilityPredictor类，传入相应的参数。
3. 训练循环：对于每个epoch，遍历训练集数据，进行前向传播、计算损失、反向传播和参数更新。关于损失函数，我想先将ground truth构建成文本形式，然后利用LLM的生成输出和ground truth文本进行对比，计算交叉熵损失。RAG使用的LLM和最终预测使用的LLM是同一个模型LLM，不加载两个LLM。RAG阶段不训练LLM，只训练最终预测阶段的LLM。
4. 验证循环：在每个epoch结束后，使用验证集数据进行模型评估，计算评估指标。评估指标包括top-K accuracy, MRR等。K取3,5,10.
5. 测试循环：在训练完成后，使用测试集数据进行最终评估。
6. 模型保存：保存训练好的模型参数和配置文件，便于后续加载和使用。
7. 日志记录：记录训练过程中的损失值、评估指标等信息，便于后续分析和调试。
8. 所有的参数通过配置文件config.yaml进行管理和传递。
---
# Prompt_3.1[完整训练流程实现]
## 背景
现在我要实现整个mobility prediction模型的训练流程。这个流程主要包括数据加载、模型初始化、训练循环、验证循环、测试循环、模型保存和日志记录等部分。

## 任务
请完成以下任务：
1. **训练流程实现**：
   - 编写/workspace/China_Journal/trainer/trainer.py和/workspace/China_Journal/dataset/dataset.py两个脚本文件。
   - 在trainer.py脚本中，实现整个训练流程，包括数据加载、模型初始化、训练循环、验证循环、测试循环、模型保存和日志记录等部分。
   - 在dataset.py脚本中，实现数据集的加载和预处理功能，确保能够正确读取处理好的训练集、验证集和测试集数据。 
2. **损失函数设计**：
   - 设计一个损失函数，先将ground truth构建成文本形式，然后利用LLM的生成输出和ground truth文本进行对比，计算交叉熵损失。
3. **评估指标实现**：
   - 实现top-K accuracy和MRR等评估指标的计算方法，K取3,5,10。

## 约束
- 使用Python编程语言。
- 使用之前编写的MobilityPredictor类和数据处理脚本。
- 训练的过程要有完善的进度显示和日志记录，记录训练过程中的损失值、评估指标等信息。
- 测试阶段，日志中记录每个阶段的中间结果，包括两个阶段的prompts,rag_summary,gravity_candidates, final_predictions等，方便调试和分析。

## 输出格式
- 提交trainer.py和dataset.py脚本文件。
- trainer.py脚本中，包括整个训练流程的实现，损失函数设计和评估指标实现。
- dataset.py脚本中，包括数据集的加载和预处理功能。
- 使用config.yaml文件管理和传递所有的参数。
- 在scripts/train.sh脚本中，添加读取config.yaml文件并传递参数给trainer.py脚本的功能。
---
# Prompt_3.2[代码优化1]
## 问题
现在我已经实现了整个mobility prediction模型的训练流程，并且进行了测试。但是在实际运行过程中，我发现有一些地方需要优化和改进：
1、config.yaml中，没有细节参数的设置，比如LLM的max_new_tokens参数、RAG中similar sample的数量、gravity的radius等。我觉得需要把这些细节参数也添加到config.yaml中，方便统一管理和传递。
2、我好像没有看到batch size的设置。我觉得batch size是一个很重要的参数，直接影响到训练的效率和效果。我希望能够在config.yaml中添加batch size的设置，并在trainer.py脚本中使用这个参数。
3、需要写一个/workspace/China_Journal/util/build_rag_database.py脚本，用于构建RAG的经验池数据库。这个脚本需要读取处理好的训练集数据，利用LLM对每个样本进行编码，生成embedding表示，并将这些embedding存储到指定路径下，作为RAG的经验池数据库。

## 任务
请完成以下任务：
1. **config.yaml优化**：
   - 修改config.yaml文件，添加LLM的max_new_tokens参数、RAG中similar sample的数量、gravity的radius和batch size等细节参数的设置。
2. **trainer.py优化**：
   - 修改trainer.py脚本，使用config.yaml中添加的batch size参数。
3. **RAG数据库构建脚本实现**：
   - 编写build_rag_database.py脚本，实现RAG经验池数据库的构建功能。
# Prompt_3.3[代码优化2]
## 背景
现在我已经实现了整个mobility prediction模型的训练流程，并且进行了测试。但是现在与信息打印的代码我觉得应该优化调整一下。我希望模型(model下的文件)代码不要打印太多的信息，主要是把信息打印的功能放在trainer.py和detailed_trainer.py脚本中，这样可以更好地控制打印的信息内容和格式。此外，我觉得config.yaml中的参数仍然不够全面，我希望能够添加更多的参数设置，方便后续的调整和优化。

## 任务
请完成以下任务：
1. **信息打印优化**：
   - 修改模型代码中的信息打印部分，模型代码中不要打印信息。
   - 在trainer.py脚本中，添加详细的信息打印功能，确保能够打印训练过程中的损失值、评估指标、训练进程等关键训练信息。
   - 在detailed_trainer.py脚本中，添加更详细的信息打印功能，确保能够打印每个阶段的中间结果，包括两个阶段的prompts, rag_summary, gravity_candidates, final_predictions等，方便调试和分析。此外，detailed_trainer.py中不使用整个数据集，而是使用一个小的子集进行快速测试和调试。
2. **config.yaml优化**：
   - 修改config.yaml文件，添加更多的参数设置，涵盖模型训练和评估的各个方面。我觉得config.yaml中的参数应该包括：
       - 数据集相关参数：城市、训练集、验证集、测试集路径，obs_len, pred_len、使用的sample数量等。
       - LLM相关参数：文件位置、max_new_tokens, temperature, top_p、top_k_predictions等。
       - RAG相关参数：similar sample的数量rag_top_m_samples, 经验池路径、rag_max_summary_length等。
       - Gravity Model相关参数：radius, weight, 候选位置数量gravity_top_n_candidates等。
       - 训练相关参数：batch size, 学习率, epoch数量, 评估指标等。
       - 其他相关参数：日志记录路径, 模型保存路径等。

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。
- Config.yaml中的参数命名要清晰明确，能够准确表达参数的含义。RAG、LLM、Gravity的参数都放在model配置项下，但要区分开来。   

## 输出格式
- 提交修改后的trainer.py和detailed_trainer.py脚本文件。
- 提交修改后的config.yaml文件。
---
# Prompt_3.4[框架结构调整：RAG模块prompt优化]
## 背景
现在我已经完成了整个mobility prediction模型的训练流程，并且进行了测试。但是在实际运行过程中，我发现LLM的回复总是充满了推理的过程，而不是直接给出我想要的答案。我觉得这是因为prompt对于返回格式设计得不够明确。我希望在RAG模块中，prompt明确指出让LLM用json格式返回rag_summary，这样可以确保LLM的回答更加结构化和易于解析。我希望修改RAG.py脚本中的prompt设计，明确指定LLM的回答格式为json格式。

## 任务
请完成以下任务：
1. **RAG模块prompt优化**：
   - 修改RAG.py脚本中的prompt设计，明确指定LLM的回答格式为json格式。
   - 确保修改后的prompt能够引导LLM生成符合json格式的rag_summary回答。
   - 在RAG.py脚本中，添加对LLM回答的解析功能，确保能够正确解析json格式的rag_summary回答。
   - 将解析出来的rag_summary加入到后续的使用流程中，确保能够正确传递和使用。

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。
- 只修改RAG相关，不修改predictor中LLM的prompt设计。

## 输出格式
- 提交修改后的RAG.py脚本文件。
---
# Prompt_3.5[BUG修复]
## 背景
现在我在测试整个trainer.py脚本的过程中，发现LLM输出Rag_summary时出现异常。训练时第一个sample是正常的，但是第二个sample，synthesis_prompt="Analyze the following mobility patterns and provide a structured summary in JSON format.\n\nQuery trajectory ends at Grid 21\n\nRetrieved 5 similar general mobility (may include walking, public transport, taxi, etc.) patterns:\n\n1. Grid 21: 5 occurrences, distance 0.20 km, similarity -inf, area type: Dining & Cusine, Life Services, time patterns: 6:00-6:00, days: Thursday\n\nYou must respond with ONLY a valid JSON object in this exact format (no additional text, explanations, or markdown):\n\n{\n  "next_locations": [\n    {\n      "grid_id": <grid_id>,\n      "frequency": <number>,\n      "distance_km": <number>,\n      "area_type": "<poi_types or \'N/A\'>",\n      "reason": "<concise reason why this location is likely>"\n    }\n  ],\n  "spatial_patterns": "<describe distance trends and area characteristics in 1-2 sentences>",\n  "temporal_patterns": "<describe time patterns in 1-2 sentences, or \'No clear temporal pattern\'>"\n}\n\nInclude top 3-5 next locations. Keep total response under 200 words."，结果LLM却一直输出感叹号。可是detailed_trainer.py脚本中使用的是同样的prompt，却没有问题。我觉得可能是trainer.py脚本中对LLM的调用方式有问题，导致LLM无法正确理解和回答prompt。我希望你能帮我找出trainer.py脚本中对LLM调用的bug，并进行修复。## 任务
请完成以下任务：
1. **BUG修复**：
   - 检查trainer.py脚本中对LLM的调用方式，找出可能导致LLM无法正确理解和回答prompt的bug。
   - 修改trainer.py脚本，修复找到的bug，确保LLM能够正确理解和回答prompt。
   - 测试修改后的trainer.py脚本，确保LLM输出的Rag_summary是正确的。
   - 底线是根据detailed_trainer.py脚本中的调用方式，调整trainer.py脚本中的调用方式，使其保持一致。
---
# Prompt_3.6.1[最终预测头和ComputeLoss函数修改]
## 背景
现在我已经完成了整个mobility prediction模型的训练流程，并且进行了测试。但是在实际运行过程中，我发现最终预测头和ComputeLoss函数需要进行一些修改和完善：
1、最终预测头现在是直接利用LLM的生成输出作为预测结果。我觉得这样可能不够准确和稳定。我希望能够在最终预测头中，添加一个简单的分类器，对LLM的生成输出进行进一步处理和优化，从而提升预测的准确性。
2、我想修改现在的预测和loss计算方式，把分类头输出的结果作为最终的预测结果。具体来说，分类头会输出每个候选位置的概率分布，然后根据这个概率分布选择top-K个位置作为最终的预测结果。这样可以更好地利用分类头的信息，提升预测的准确性。

## 任务
请完成以下任务：
1. **最终预测头修改**：
   - 修改MobilityPredictor.py脚本中的最终预测头部分，添加一个简单的分类器，对LLM的生成输出进行进一步处理和优化。
2. **ComputeLoss函数修改**：
   - 修改trainer.py脚本中的ComputeLoss函数，使用分类头的输出结果作为最终的预测结果。
   - 设计一个新的损失计算方式，基于分类头输出的概率分布，计算交叉熵损失。
3. **测试修改**：
   - 测试修改后的MobilityPredictor.py和trainer.py脚本，确保最终预测头和ComputeLoss函数能够正确工作，并提升预测的准确性。

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- trainer.py中不要太多的打印。

## 输出格式
- 提交修改后的MobilityPredictor.py和trainer.py脚本文件。
---
# Prompt_3.6.2[最终预测头和ComputeLoss函数修改V2]
## 背景
我之前修改了MobilityPredictor.py和trainer.py脚本，添加了一个分类器作为最终预测头，并修改了ComputeLoss函数来使用分类头的输出结果作为最终的预测结果。但是我现在还是想修改为不要分类头、LLM直接输出的方式。我觉得这样可以更好地利用LLM的强大生成能力，提升预测的准确性。我想修改修改prompt，最后不需要输出多个候选位置，而是直接输出最终的预测结果。损失函数方面，我想修改为直接使用LLM的生成输出和模板化的ground truth文本进行对比，计算交叉熵损失。在测试阶段，我希望传入num_beam=top_k_predictions，这样可以让LLM生成多个候选结果，提升预测的多样性和准确性。
## 任务
请完成以下任务：
1. **最终预测头修改**：
   - 修改MobilityPredictor.py脚本中的最终预测头部分，移除分类器，改为直接使用LLM的生成输出作为最终的预测结果。
2. **ComputeLoss函数修改**：
   - 修改trainer.py脚本中的ComputeLoss函数，改为使用LLM的生成输出作为最终的预测结果。将ground_truth先通过模板变成text，例如“”The next locations is Grid 34”，然后计算LLM生成结果与的交叉熵损失。让LLM训练完后，输出的结果尽量符合这个模板。
3. **Prompt修改**
   - 修改MobilityPredictor.py脚本中的最终预测prompt设计，确保LLM直接输出最终的预测结果，而不是多个候选位置。
4. **测试修改**：
   - 测试修改后的MobilityPredictor.py和trainer.py脚本，确保最终预测头和ComputeLoss函数能够正确工作，并提升预测的准确性。

## 约束  
- 使用Python编程语言。
- 删除原来分类头的相关代码，保持代码整洁。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- trainer.py中不要太多的打印。
- 训练阶段只有一个ground_truth,但是测试阶段，传入num_beam=top_k_predictions，预测多个可能结果。仍然需要计算top-k accuracy，mrr。
## 输出格式
- 提交修改后的MobilityPredictor.py和trainer.py脚本文件。
---


# Rag库建立
我目前的想法是;
1、读取train.csv。
2、设计一个prompt，“This is a daily mobility trajectory.Summarize its mobility pattern in terms of purpose, temporal rhythm, and functional transitions.”，将train.csv中的每个样本的历史轨迹数据输入到LLM中，生成embedding表示。
3、将membedding归一化、PCA降维、HDBSCAN聚类，得到高频通勤簇和低频但结构鲜明的特殊出行簇，一共分为1k个类。注意这个归一化的操作仅仅是用于聚类，后续的存储rag库和相似度计算仍然使用原始embedding。
4、根据聚类的结果每个类别选择一些代表样本，组成最终的rag_database，保存到/workspace/China_Journal/util/rag_database路径下。选择的类别包括中心原型、边缘样本、context或时间极端样本。每个类别选择10个样本（不够就不需要10个），最终组成1k*10=10k个样本的rag_database。
5、要加一个“功能性覆盖约束”，使得rag_database能够有较好的起点功能覆盖、时间段覆盖、空间覆盖和跨区距离等级覆盖。（尽量，不是必须）
# Prompt_4.1[RAG经验池数据库构建脚本实现]
## 背景
现在我需要建立RAG的经验池数据库，用于后续的相似度搜索和summary生成。这个数据库需要包含处理好的训练集数据的embedding表示，存储在/workspace/China_Journal/util/rag_database路径下。我需要一个Python脚本来实现这个功能。具体来说用如下的步骤：
1、读取train.csv。
2、设计一个prompt，“This is a daily mobility trajectory.Summarize its mobility pattern in terms of purpose, temporal rhythm, and functional transitions.”，将train.csv中的每个样本的历史轨迹数据输入到LLM中，生成embedding表示。
3、将membedding归一化、PCA降维、HDBSCAN聚类，得到高频通勤簇和低频但结构鲜明的特殊出行簇，一共分为1k个类。注意这个归一化的操作仅仅是用于聚类，后续的存储rag库和相似度计算仍然使用原始embedding。
4、根据聚类的结果每个类别选择一些代表样本，组成最终的rag_database，保存到/workspace/China_Journal/util/rag_database路径下。选择的类别包括中心原型、边缘样本、context或时间极端样本。每个类别选择10个样本（不够就不需要10个），最终组成1k*10=10k个样本的rag_database。
5、要加一个“功能性覆盖约束”，使得rag_database能够有较好的起点功能覆盖、时间段覆盖、空间覆盖和跨区距离等级覆盖。（尽量，不是必须）

## 任务
请完成以下任务：
1. **RAG数据库构建脚本实现**：
   - 编写/workspace/China_Journal/util/build_rag_database.py脚本，实现RAG经验池数据库的构建功能。
   - 脚本需要按照上述步骤实现，包括读取train.csv、生成embedding表示、归一化和聚类、选择代表样本、保存rag_database等部分。
2. **功能性覆盖约束实现**：
   - 在选择代表样本的过程中，尽量考虑功能性覆盖约束，确保rag_database能够有较好的起点功能覆盖、时间段覆盖、空间覆盖和跨区距离等级覆盖。

## 约束
- 使用Python编程语言。
- 使用之前编写的LLM.py脚本中的类和函数。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。
- 输出的rag_database格式要与RAG.py脚本中的读取方式保持一致，确保能够正确加载和使用。
## 输出格式
- 提交build_rag_database.py脚本文件。
- 脚本中，包括RAG经验池数据库的构建功能和功能性覆盖约束的实现。
---

# 系统模型优化
# Prompt_5.1[系统模型优化-Prompt优化]
## 背景
现在我已经完成了整个mobility prediction模型的训练流程，并且进行了测试。但是在实际运行过程中，我觉得prompt还可以进行一些优化和改进，以提升模型的性能和预测的准确性。修改之后我想先用detailed_trainer.py脚本进行测试，确保修改后的prompt能够提升RAG_summary的质量。
## 任务
请完成以下任务：
1. **RAG模块prompt优化**：
   - RAG.py脚本中的_generate_structured_summary函数中，利用了dist，但是计算的是当前位置和similar sample中最后一个位置的距离。我觉得这样可能不够准确。我希望改为计算当前位置和similar sample中倒数第二个位置的距离。因为similar sample的最后一个位置是下一个位置，和当前轨迹没有空间关系，计算距离没有意义。我希望通过similar sample中倒数第二个位置和当前轨迹的距离，来反映空间关系，为现在下一步预测提供更有用的信息。
   - 使用如下的prompt设计：
   ```synthesis_prompt = f"""
You are given a set of retrieved mobility trajectories that are semantically similar to a query trajectory.
These trajectories serve as supporting evidence and may be noisy or incomplete.

Your task is to summarize ONLY the patterns that are consistently observed across the retrieved samples.
Do NOT introduce external knowledge, assumptions, or speculation beyond the given data.

{context}

You must respond with ONLY a valid JSON object in the following exact format.
Do not include explanations, markdown, or any text outside the JSON.

{{
  "next_locations": [
    {{
      "grid_id": <grid_id>,
      "frequency": <integer count in retrieved samples>,
      "avg_distance_km": <average distance from previous location>,
      "area_type": "<dominant POI categories or 'N/A'>",
      "evidence_basis": "<which observed factors support this candidate, e.g. time similarity, POI transition, distance range>"
    }}
  ],
  "spatial_patterns": {{
    "distance_range_km": "<typical min–max distance or 'inconsistent'>",
    "movement_type": "<short-range | medium-range | long-range | mixed>"
  }},
  "temporal_patterns": {{
    "dominant_time_windows": "<e.g. morning_peak, evening, mixed, or 'none'>",
    "temporal_consistency": "<high | medium | low>"
  }},
  "pattern_confidence": "<high | medium | low>"
}}

Rules:
- Include only the top 3–5 next locations by frequency.
- All fields must be derived from the retrieved samples.
- If no clear pattern exists, explicitly state 'inconsistent' or 'none'.
- Keep the total response concise and factual.
"""
   - 确保修改后的prompt能够引导LLM生成符合json格式的rag_summary回答。
2. **修改detailed_trainer.py**
   - 修改detailed_trainer.py脚本中的RAG模块调用部分，使用上述新的prompt设计。
   - 确保修改后的detailed_trainer.py脚本能够正确调用RAG模块，并生成符合新prompt要求的rag_summary回答。

## 约束
- 使用Python编程语言。
- 保持原有脚本的结构和逻辑，尽量只修改必要的部分。
- 过程中需要有适当的注释和过程打印。
- 只修改RAG相关，不修改predictor中LLM的prompt设计。
## 输出格式
- 提交修改后的RAG.py和detailed_trainer.py脚本文件。
- RAG.py脚本中，包括RAG模块prompt优化的实现。
- detailed_trainer.py脚本中，包括RAG模块调用部分的修改。
---


