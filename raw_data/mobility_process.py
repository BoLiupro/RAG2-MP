import pandas as pd
import numpy as np
import os
import argparse
from typing import Tuple, List, Dict
import glob
from datetime import datetime, timedelta
from tqdm import tqdm
import json

# ========== 坐标转换函数 (复用 poi_process.py) ==========
def gcj02_to_wgs84(lng: float, lat: float) -> Tuple[float, float]:
    """
    GCJ-02(火星坐标系)转WGS-84
    """
    a = 6378245.0
    ee = 0.00669342162296594323
    
    def transform_lat(lng, lat):
        ret = -100.0 + 2.0 * lng + 3.0 * lat + 0.2 * lat * lat + \
              0.1 * lng * lat + 0.2 * np.sqrt(np.abs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lat * np.pi) + 40.0 *
                np.sin(lat / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (160.0 * np.sin(lat / 12.0 * np.pi) + 320 *
                np.sin(lat * np.pi / 30.0)) * 2.0 / 3.0
        return ret
    
    def transform_lng(lng, lat):
        ret = 300.0 + lng + 2.0 * lat + 0.1 * lng * lng + \
              0.1 * lng * lat + 0.1 * np.sqrt(np.abs(lng))
        ret += (20.0 * np.sin(6.0 * lng * np.pi) + 20.0 *
                np.sin(2.0 * lng * np.pi)) * 2.0 / 3.0
        ret += (20.0 * np.sin(lng * np.pi) + 40.0 *
                np.sin(lng / 3.0 * np.pi)) * 2.0 / 3.0
        ret += (150.0 * np.sin(lng / 12.0 * np.pi) + 300.0 *
                np.sin(lng / 30.0 * np.pi)) * 2.0 / 3.0
        return ret
    
    dlat = transform_lat(lng - 105.0, lat - 35.0)
    dlng = transform_lng(lng - 105.0, lat - 35.0)
    radlat = lat / 180.0 * np.pi
    magic = np.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = np.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * np.pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * np.cos(radlat) * np.pi)
    mglat = lat + dlat
    mglng = lng + dlng
    return lng * 2 - mglng, lat * 2 - mglat

def assign_grid_id(lng: float, lat: float, bbox: Tuple[float, float, float, float], 
                   grid_size: int = 40) -> int:
    """
    根据坐标分配网格ID
    """
    min_lng, min_lat, max_lng, max_lat = bbox
    
    # 检查是否在研究区域内
    if lng < min_lng or lng > max_lng or lat < min_lat or lat > max_lat:
        return -1
    
    lng_step = (max_lng - min_lng) / grid_size
    lat_step = (max_lat - min_lat) / grid_size
    
    i = int((lng - min_lng) / lng_step)
    j = int((lat - min_lat) / lat_step)
    
    # 处理边界情况
    if i >= grid_size:
        i = grid_size - 1
    if j >= grid_size:
        j = grid_size - 1
    
    return i * grid_size + j

# ========== 数据处理类 ==========
class MobilityProcessor:
    def __init__(self, city_name: str, bbox: Tuple[float, float, float, float], 
                 grid_size: int = 40, time_interval_hours: float = 0.5,
                 obs_len: int = 12, pred_len: int = 1,
                 split_ratios: List[float] = [0.4, 0.4, 0.1, 0.1],
                 max_days: int = None):
        self.city_name = city_name
        self.bbox = bbox
        self.grid_size = grid_size
        self.time_interval = timedelta(hours=time_interval_hours)
        self.obs_len = obs_len
        self.pred_len = pred_len
        self.split_ratios = split_ratios  # [rag, train, val, test]
        self.max_days = max_days  # 限制处理的天数
        
    def process_nanchang(self, data_dir: str) -> pd.DataFrame:
        print("正在处理南昌数据...")
        # 1. 读取Location数据
        loc_file = os.path.join(data_dir, 'location.csv')
        print(f"读取Location文件: {loc_file}")
        loc_df = pd.read_csv(loc_file)
        # 创建 location_id -> (lng, lat) 映射
        loc_map = loc_df.set_index('base_id')[['longitude', 'latitude']].to_dict('index')
        
        # 2. 读取App Usage数据
        files = glob.glob(os.path.join(data_dir, 'app_usage_record_*.csv'))
        print(f"找到 {len(files)} 个应用使用记录文件")
        
        all_data = []
        for f in files:
            print(f"读取: {os.path.basename(f)}")
            df = pd.read_csv(f)
            # 只需要 user_id, time, location
            df = df[['user_id', 'time', 'location']]
            all_data.append(df)
            
        full_df = pd.concat(all_data, ignore_index=True)
        print(f"原始数据总条数: {len(full_df)}")
        
        # 3. 映射经纬度
        print("映射经纬度...")
        def get_coords(loc_id):
            if loc_id in loc_map:
                return loc_map[loc_id]['longitude'], loc_map[loc_id]['latitude']
            return None, None
            
        coords = full_df['location'].apply(get_coords)
        full_df['lng'] = [c[0] for c in coords]
        full_df['lat'] = [c[1] for c in coords]
        
        # 过滤掉找不到坐标的记录
        full_df.dropna(subset=['lng', 'lat'], inplace=True)
        
        # 4. 转换时间格式 YYYYMMDDHHMMSS -> datetime
        print("转换时间格式...")
        full_df['timestamp'] = pd.to_datetime(full_df['time'], format='%Y%m%d%H%M%S')
        
        return full_df[['user_id', 'timestamp', 'lng', 'lat']]

    def process_beijing(self, data_dir: str) -> pd.DataFrame:
        print("正在处理北京数据...")
        file_path = os.path.join(data_dir, 'uid2info.json')
        print(f"读取文件: {file_path}")
        
        with open(file_path, 'r') as f:
            data = json.load(f)
        
        print(f"用户数量: {len(data)}")
        
        all_data = []
        for user_entry in tqdm(data, desc="Processing users"):
            user_id = user_entry['user_id']
            traces = user_entry['user_traces']
            
            for trace in traces:
                # trace format: [timestamp_str, poi_category, lng, lat]
                # timestamp_str format: MMDDHHMM (e.g., "07070904" = July 7, 09:04)
                timestamp_str = trace[0]
                lng = trace[2]
                lat = trace[3]
                
                # Parse timestamp: MMDDHHMM -> datetime
                # Assuming year 2022 (based on POI data date)
                month = int(timestamp_str[:2])
                day = int(timestamp_str[2:4])
                hour = int(timestamp_str[4:6])
                minute = int(timestamp_str[6:8])
                
                try:
                    timestamp = datetime(2022, month, day, hour, minute)
                    all_data.append({
                        'user_id': user_id,
                        'timestamp': timestamp,
                        'lng': lng,
                        'lat': lat
                    })
                except ValueError:
                    # Skip invalid dates
                    continue
        
        full_df = pd.DataFrame(all_data)
        print(f"原始数据总条数: {len(full_df)}")
        return full_df

    def process_shenzhen(self, data_dir: str) -> pd.DataFrame:
        print("正在处理深圳数据...")
        file_path = os.path.join(data_dir, 'shenzhen_202208.csv')
        print(f"读取文件: {file_path}")
        df = pd.read_csv(file_path)
        
        # 深圳数据: ObjectID(user), StartTime, StartLon, StartLat, StopTime, StopLon, StopLat
        # 将每条记录拆分为起点和终点两个点
        print("拆分起终点...")
        start_df = df[['ObjectID', 'StartTime', 'StartLon', 'StartLat']].copy()
        start_df.columns = ['user_id', 'timestamp', 'lng', 'lat']
        
        stop_df = df[['ObjectID', 'StopTime', 'StopLon', 'StopLat']].copy()
        stop_df.columns = ['user_id', 'timestamp', 'lng', 'lat']
        
        full_df = pd.concat([start_df, stop_df], ignore_index=True)
        
        # 转换时间格式
        full_df['timestamp'] = pd.to_datetime(full_df['timestamp'])
        
        print(f"原始数据总条数: {len(full_df)}")
        return full_df

    def discretize_and_sample(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        print("开始离散化和采样...")
        
        # 1. 坐标转换 (GCJ02 -> WGS84)
        # 用户提示: mobility的数据本身就基于WGS-84坐标系了，无需转换
        
        # 2. 分配网格ID (向量化优化)
        print("分配网格ID...")
        min_lng, min_lat, max_lng, max_lat = self.bbox
        lng_step = (max_lng - min_lng) / self.grid_size
        lat_step = (max_lat - min_lat) / self.grid_size
        
        # 过滤不在bbox内的点
        in_bbox = (df['lng'] >= min_lng) & (df['lng'] <= max_lng) & \
                  (df['lat'] >= min_lat) & (df['lat'] <= max_lat)
        df = df[in_bbox].copy()
        
        # 计算网格坐标
        rows = ((df['lng'] - min_lng) / lng_step).astype(int)
        cols = ((df['lat'] - min_lat) / lat_step).astype(int)
        
        # 边界处理 (clip)
        rows = np.clip(rows, 0, self.grid_size - 1)
        cols = np.clip(cols, 0, self.grid_size - 1)
        
        df['location_id'] = rows * self.grid_size + cols
        
        print(f"网格内有效数据条数: {len(df)}")
        
        # 3. 按用户分组处理
        print("按用户分组处理轨迹...")
        users = df['user_id'].unique()
        np.random.shuffle(users) # 随机打乱以便划分数据集
        
        # 划分用户集 (rag:train:val:test = split_ratios)
        n_users = len(users)
        rag_ratio, train_ratio, val_ratio, test_ratio = self.split_ratios
        
        rag_end = int(rag_ratio * n_users)
        train_end = int((rag_ratio + train_ratio) * n_users)
        val_end = int((rag_ratio + train_ratio + val_ratio) * n_users)
        
        rag_users = set(users[:rag_end])
        train_users = set(users[rag_end:train_end])
        val_users = set(users[train_end:val_end])
        test_users = set(users[val_end:])
        
        print(f"用户划分 ({rag_ratio*100:.0f}%:{train_ratio*100:.0f}%:{val_ratio*100:.0f}%:{test_ratio*100:.0f}%): "
              f"RAG={len(rag_users)}, Train={len(train_users)}, Val={len(val_users)}, Test={len(test_users)}")
        
        rag_samples = []
        train_samples = []
        val_samples = []
        test_samples = []
        
        # 预计算时间间隔
        interval_seconds = self.time_interval.total_seconds()
        
        # 使用 tqdm 显示进度
        for uid, group in tqdm(df.groupby('user_id'), desc="Processing users"):
            group = group.sort_values('timestamp')
            
            if len(group) < 2:
                continue
                
            # 时间离散化
            # 获取该用户的时间范围
            min_time = group['timestamp'].min()
            max_time = group['timestamp'].max()
            
            # 向下取整到最近的时间间隔
            start_time = min_time - timedelta(
                seconds=min_time.timestamp() % interval_seconds)
            end_time = max_time - timedelta(
                seconds=max_time.timestamp() % interval_seconds)
                
            # 生成完整的时间序列
            time_range = pd.date_range(start=start_time, end=end_time, freq=self.time_interval)
            
            # 将原始数据映射到时间槽
            group['time_slot'] = group['timestamp'].apply(
                lambda t: t - timedelta(seconds=t.timestamp() % interval_seconds))
            
            # 每个时间槽取第一个点
            resampled = group.groupby('time_slot').first().reindex(time_range)
            
            # 填充缺失值 (Forward Fill)
            resampled['location_id'] = resampled['location_id'].ffill()
            resampled['user_id'] = uid
            
            # 移除开头可能的NaN (如果第一个时间槽没有数据)
            resampled.dropna(subset=['location_id'], inplace=True)
            
            if len(resampled) < self.obs_len + self.pred_len:
                continue
                
            # 滑动窗口生成样本
            # 这里我们不直接生成 (X, Y) 矩阵，而是生成符合要求的轨迹序列CSV
            # 题目要求: "每个user按照obs_len+pred_len进行滑动窗口切片...符合的保留"
            # 输出格式: user_id, timestamp, location_id
            # 实际上，如果我们要输出CSV，通常是输出处理后的完整轨迹，或者切片后的样本。
            # 题目示例: "每个csv文件包含user_id, timestamp, location_id... timestamp是离散化后的时间戳"
            # 这听起来像是输出离散化后的轨迹，而不是切片后的样本矩阵。
            # 但是题目又说 "所有samples进行检查...符合的保留"。
            # 如果我们只输出轨迹，那么后续模型训练时再切片？
            # 题目要求 "生成处理后的出行轨迹数据csv文件...包括train.csv, val.csv, test.csv"
            # 且 "每个csv文件包含user_id, timestamp, location_id"
            # 这通常意味着输出的是清洗、离散化后的轨迹数据。
            # 至于 "滑动窗口切片...检查...符合的保留"，这可能是指：
            # 如果一个用户的轨迹中间有一段长时间缺失（ffill太久），或者切片后发现有不在网格内的点（虽然我们已经过滤了），
            # 那么这段轨迹应该被丢弃。
            # 由于我们已经过滤了不在网格内的点，且使用了ffill，
            # 唯一的问题是如果ffill跨度太大是否合理？题目说 "中间缺失的位置可以用前一个位置进行填充"，没说限制。
            # 所以我将输出离散化后的完整轨迹。
            
            # 转换 location_id 为 int
            resampled['location_id'] = resampled['location_id'].astype(int)
            
            # 格式化时间戳
            resampled['timestamp_str'] = resampled.index.strftime('%Y%m%d %H:%M')
            
            # 准备输出数据
            out_data = resampled[['user_id', 'location_id', 'timestamp_str']].reset_index()
            out_data.rename(columns={'index': 'timestamp_dt'}, inplace=True)
            # 保留datetime列用于后续排序，timestamp_str用于输出
            out_data = out_data[['user_id', 'timestamp_dt', 'timestamp_str', 'location_id']]
            
            if uid in rag_users:
                rag_samples.append(out_data)
            elif uid in train_users:
                train_samples.append(out_data)
            elif uid in val_users:
                val_samples.append(out_data)
            elif uid in test_users:
                test_samples.append(out_data)
        
        # 合并并排序数据 - 确保时间连续性
        print("\n合并并排序数据...")
        
        def finalize_dataset(samples):
            if not samples:
                return pd.DataFrame()
            df = pd.concat(samples, ignore_index=True)
            # 按用户ID和时间戳排序，确保每个用户的轨迹是时间连续的
            df = df.sort_values(['user_id', 'timestamp_dt'])
            # 只保留需要的列用于输出
            df = df[['user_id', 'timestamp_str', 'location_id']].rename(columns={'timestamp_str': 'timestamp'})
            return df
        
        return (
            finalize_dataset(rag_samples),
            finalize_dataset(train_samples),
            finalize_dataset(val_samples),
            finalize_dataset(test_samples)
        )
    
    def filter_by_days(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        按天数过滤数据，保留最近的N天数据
        
        Args:
            df: 输入数据集（必须包含timestamp列）
            
        Returns:
            过滤后的数据集
        """
        if df.empty or self.max_days is None:
            return df
            
        print(f"\n按天数过滤数据 (保留最近 {self.max_days} 天)...")
        
        # 找到最大日期
        max_date = df['timestamp'].max()
        # 计算截止日期
        cutoff_date = max_date - timedelta(days=self.max_days)
        
        # 过滤数据
        filtered_df = df[df['timestamp'] >= cutoff_date].copy()
        
        original_records = len(df)
        filtered_records = len(filtered_df)
        original_date_range = (df['timestamp'].max() - df['timestamp'].min()).days
        filtered_date_range = (filtered_df['timestamp'].max() - filtered_df['timestamp'].min()).days
        
        print(f"  原始数据: {original_records:,} 条记录, 时间跨度: {original_date_range} 天")
        print(f"  过滤后: {filtered_records:,} 条记录 ({filtered_records/original_records*100:.1f}%), "
              f"时间跨度: {filtered_date_range} 天")
        print(f"  日期范围: {filtered_df['timestamp'].min()} 至 {filtered_df['timestamp'].max()}")
        
        return filtered_df

def main():
    parser = argparse.ArgumentParser(description='出行轨迹数据处理脚本')
    parser.add_argument('--city', type=str, required=True,
                        choices=['shenzhen', 'nanchang', 'beijing'],
                        help='要处理的城市')
    parser.add_argument('--data_dir', type=str, default='.',
                        help='数据目录路径')
    parser.add_argument('--interval', type=float, default=1,
                        help='时间间隔(小时)，如0.25(15分钟), 0.5(30分钟), 1.0(60分钟)')
    parser.add_argument('--obs_len', type=int, default=12,
                        help='观察长度')
    parser.add_argument('--pred_len', type=int, default=1,
                        help='预测长度')
    parser.add_argument('--split_ratios', type=float, nargs=4, 
                        default=[0.45, 0.45, 0.05, 0.05],
                        help='数据集划分比例 [rag, train, val, test], 默认: 0.45 0.45 0.05 0.05')
    parser.add_argument('--max_days', type=int, default=5,
                        help='限制处理的天数（保留最近N天数据），默认None表示处理所有天数')
    
    args = parser.parse_args()
    
    # 验证split_ratios总和为1
    if abs(sum(args.split_ratios) - 1.0) > 0.01:
        raise ValueError(f"split_ratios的总和必须为1.0，当前为{sum(args.split_ratios)}")
    
    # 验证max_days
    if args.max_days is not None and args.max_days <= 0:
        raise ValueError(f"max_days必须大于0，当前为{args.max_days}")
    
    # 城市配置
    CITY_CONFIGS = {
        'shenzhen': {
            'bbox': (114.02, 22.51, 114.18, 22.67),
            'sub_dir': 'shenzhen'
        },
        'nanchang': {
            'bbox': (115.78, 28.58, 115.94, 28.74),
            'sub_dir': 'nanchang'
        },
        'beijing': {
            'bbox': (116.31, 39.84, 116.47, 40.00),
            'sub_dir': 'beijing'
        }
    }
    
    config = CITY_CONFIGS[args.city]
    data_path = os.path.join(args.data_dir, config['sub_dir'])
    
    processor = MobilityProcessor(
        city_name=args.city,
        bbox=config['bbox'],
        time_interval_hours=args.interval,
        obs_len=args.obs_len,
        pred_len=args.pred_len,
        split_ratios=args.split_ratios,
        max_days=args.max_days
    )
    
    # 1. 加载数据
    if args.city == 'nanchang':
        df = processor.process_nanchang(data_path)
    elif args.city == 'beijing':
        df = processor.process_beijing(data_path)
    else:
        df = processor.process_shenzhen(data_path)
    
    # 2. 按天数过滤数据（如果指定了max_days）
    if args.max_days is not None:
        df = processor.filter_by_days(df)
        
    # 3. 处理数据 (离散化和数据集划分)
    rag_df, train_df, val_df, test_df = processor.discretize_and_sample(df)
    
    # 4. 保存结果到 data/{city} 目录
    # 计算正确的输出目录：从raw_data目录回到项目根目录，然后进入data/{city}
    script_dir = os.path.dirname(os.path.abspath(__file__))  # raw_data目录
    project_root = os.path.dirname(script_dir)  # 项目根目录
    output_dir = os.path.join(project_root, 'data', args.city)
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"\n保存结果到 {output_dir} ...")
    if not rag_df.empty:
        rag_df.to_csv(os.path.join(output_dir, 'rag.csv'), index=False)
        print(f"  ✓ rag.csv: {len(rag_df):,} 条记录")
    if not train_df.empty:
        train_df.to_csv(os.path.join(output_dir, 'train.csv'), index=False)
        print(f"  ✓ train.csv: {len(train_df):,} 条记录")
    if not val_df.empty:
        val_df.to_csv(os.path.join(output_dir, 'val.csv'), index=False)
        print(f"  ✓ val.csv: {len(val_df):,} 条记录")
    if not test_df.empty:
        test_df.to_csv(os.path.join(output_dir, 'test.csv'), index=False)
        print(f"  ✓ test.csv: {len(test_df):,} 条记录")
    
    # 5. 打印详细统计信息
    print(f"\n{'='*70}")
    print(f"数据集统计 - {args.city.upper()}")
    print(f"{'='*70}")
    
    # 5.1 用户数量统计
    rag_users = rag_df['user_id'].nunique() if not rag_df.empty else 0
    train_users = train_df['user_id'].nunique() if not train_df.empty else 0
    val_users = val_df['user_id'].nunique() if not val_df.empty else 0
    test_users = test_df['user_id'].nunique() if not test_df.empty else 0
    total_users = rag_users + train_users + val_users + test_users
    
    print(f"\n[用户数量统计]")
    print(f"  RAG数据集: {rag_users} 用户 ({rag_users/total_users*100:.1f}%)")
    print(f"  训练集: {train_users} 用户 ({train_users/total_users*100:.1f}%)")
    print(f"  验证集: {val_users} 用户 ({val_users/total_users*100:.1f}%)")
    print(f"  测试集: {test_users} 用户 ({test_users/total_users*100:.1f}%)")
    print(f"  总计: {total_users} 用户")
    
    # 5.2 记录数量统计
    rag_records = len(rag_df)
    train_records = len(train_df)
    val_records = len(val_df)
    test_records = len(test_df)
    total_records = rag_records + train_records + val_records + test_records
    
    print(f"\n[记录数量统计]")
    print(f"  RAG数据集: {rag_records:,} 条记录 ({rag_records/total_records*100:.1f}%)")
    print(f"  训练集: {train_records:,} 条记录 ({train_records/total_records*100:.1f}%)")
    print(f"  验证集: {val_records:,} 条记录 ({val_records/total_records*100:.1f}%)")
    print(f"  测试集: {test_records:,} 条记录 ({test_records/total_records*100:.1f}%)")
    print(f"  总计: {total_records:,} 条记录")
    
    # 5.3 空间分布统计
    all_df = pd.concat([rag_df, train_df, val_df, test_df], ignore_index=True)
    unique_locations = all_df['location_id'].nunique()
    
    print(f"\n[空间分布统计]")
    print(f"  研究区域: {config['bbox']}")
    print(f"  网格总数: {processor.grid_size * processor.grid_size} ({processor.grid_size}x{processor.grid_size})")
    print(f"  覆盖网格数: {unique_locations}")
    print(f"  网格覆盖率: {unique_locations/(processor.grid_size*processor.grid_size)*100:.1f}%")
    
    # 4.4 时间分布统计
    all_df['timestamp'] = pd.to_datetime(all_df['timestamp'], format='%Y%m%d %H:%M')
    
    print(f"\n[时间分布统计]")
    print(f"  开始时间: {all_df['timestamp'].min()}")
    print(f"  结束时间: {all_df['timestamp'].max()}")
    print(f"  时间跨度: {(all_df['timestamp'].max() - all_df['timestamp'].min()).days} 天")
    print(f"  时间间隔: {args.interval} 小时 ({args.interval*60:.0f} 分钟)")
    
    # 5.5 数据集划分比例确认
    print(f"\n[数据集划分比例]")
    print(f"  目标比例: {args.split_ratios[0]*100:.0f}% : {args.split_ratios[1]*100:.0f}% : "
          f"{args.split_ratios[2]*100:.0f}% : {args.split_ratios[3]*100:.0f}%")
    print(f"  实际比例(用户): {rag_users/total_users*100:.1f}% : {train_users/total_users*100:.1f}% : "
          f"{val_users/total_users*100:.1f}% : {test_users/total_users*100:.1f}%")
    print(f"  实际比例(记录): {rag_records/total_records*100:.1f}% : {train_records/total_records*100:.1f}% : "
          f"{val_records/total_records*100:.1f}% : {test_records/total_records*100:.1f}%")
    
    # 5.6 天数过滤统计
    if args.max_days is not None:
        print(f"\n[天数过滤统计]")
        print(f"  限制天数: {args.max_days} 天")
        print(f"  过滤后总记录数: {total_records:,}")
    else:
        print(f"\n[天数过滤统计]")
        print(f"  未进行天数过滤 (max_days = None)")
    
    print(f"\n{'='*70}")
    print("处理完成！")
    print(f"{'='*70}\n")

if __name__ == '__main__':
    main()
