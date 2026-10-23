import streamlit as st
import pandas as pd
import folium
from streamlit_folium import st_folium
import networkx as nx
import torch
from data_models import Vehicle, FireStation, WaterSource, FireEvent, WATER_DEMAND
from road_network import (
    build_road_network,
    get_node_coords,
    get_poi_nodes,
    get_gcj_coords,
    haversine,
    get_scene_weight_multiplier,
    OSM_COORDS,
    SPEED
)
from one_vehicle import dijkstra_weighted_path
from space_time import SpaceTimeGrid
from many_vehicle import schedule_relay
from dqn_scheduler import DQNScheduler
from data_fusion import MultiModalDataFusion

st.set_page_config(page_title="特种车辆动态路径优化与评估平台", page_icon="🚒", layout="wide")
def extract_subgraph_by_distance(G, center_lat, center_lon, radius_m=1500):
    """
    根据中心点经纬度（WGS84）和半径（米）提取子图
    返回原图的子图（只保留距离 ≤ radius_m 的节点及其之间的边）
    """
    nodes_to_keep = []
    for node, data in G.nodes(data=True):
        pos = data.get('pos')
        if pos is None or len(pos) != 2:
            continue
        try:
            node_lat, node_lon = float(pos[0]), float(pos[1])
            dist = haversine(center_lat, center_lon, node_lat, node_lon)
            if dist <= radius_m:
                nodes_to_keep.append(node)
        except (TypeError, ValueError):
            continue
    subG = G.subgraph(nodes_to_keep).copy()
    print(f"路网裁剪：原节点 {G.number_of_nodes()} → {subG.number_of_nodes()}，边 {G.number_of_edges()} → {subG.number_of_edges()}")
    return subG
# 缓存路网构建
@st.cache_resource
def get_road_network():
    G = build_road_network()
    # 以着火点为中心，半径1000米裁剪路网
    center_lat, center_lon = OSM_COORDS["盛世荷园小区"]  # (28.210667, 112.977931)
    G = extract_subgraph_by_distance(G, center_lat, center_lon, radius_m=1500)
    return G

try:
    G = get_road_network()
except Exception as e:
    st.error(f"路网加载失败：{e}")
    st.stop()

NODE_COORDS_WGS = get_node_coords(G)
POI_NODES = get_poi_nodes(G)

# 检查关键POI是否在1km范围内
required_pois = ["盛世荷园小区", "清水塘街道消防站", "潮宗街政府专职消防救援站", "湘雅路街道文昌阁消防站"]
missing = [p for p in required_pois if p not in POI_NODES]
if missing:
    st.warning(f"以下POI超出1km路网范围，可能影响调度：{missing}")
    # 如果火场本身不在子图中，则无法继续
    if "盛世荷园小区" not in POI_NODES:
        st.error("着火点不在路网范围内，请增大裁剪半径或检查坐标")
        st.stop()
GCJ_COORDS = get_gcj_coords()

if not POI_NODES:
    st.error("POI 节点映射失败")
    st.stop()

FIRE_NODE = POI_NODES.get("盛世荷园小区")
if FIRE_NODE is None:
    st.error("火场节点未映射")
    st.stop()
WATER_SOURCE_NODE = POI_NODES.get("独立供水点")

fire_gcj = GCJ_COORDS["盛世荷园小区"]  # (lon, lat)
FIRE_GCJ_LOC = (fire_gcj[1], fire_gcj[0])  # (lat, lon) for folium

def wgs84_to_gcj02(lon, lat):
    import math
    pi = 3.14159265358979324
    a = 6378245.0
    ee = 0.00669342162296594323
    def transform_lat(x, y):
        ret = -100.0 + 2.0 * x + 3.0 * y + 0.2 * y * y + 0.1 * x * y + 0.2 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(y * pi) + 40.0 * math.sin(y / 3.0 * pi)) * 2.0 / 3.0
        ret += (160.0 * math.sin(y / 12.0 * pi) + 320 * math.sin(y * pi / 30.0)) * 2.0 / 3.0
        return ret
    def transform_lng(x, y):
        ret = 300.0 + x + 2.0 * y + 0.1 * x * x + 0.1 * x * y + 0.1 * math.sqrt(abs(x))
        ret += (20.0 * math.sin(6.0 * x * pi) + 20.0 * math.sin(2.0 * x * pi)) * 2.0 / 3.0
        ret += (20.0 * math.sin(x * pi) + 40.0 * math.sin(x / 3.0 * pi)) * 2.0 / 3.0
        ret += (150.0 * math.sin(x / 12.0 * pi) + 300.0 * math.sin(x / 30.0 * pi)) * 2.0 / 3.0
        return ret
    dlat = transform_lat(lon - 105.0, lat - 35.0)
    dlng = transform_lng(lon - 105.0, lat - 35.0)
    radlat = lat / 180.0 * pi
    magic = math.sin(radlat)
    magic = 1 - ee * magic * magic
    sqrtmagic = math.sqrt(magic)
    dlat = (dlat * 180.0) / ((a * (1 - ee)) / (magic * sqrtmagic) * pi)
    dlng = (dlng * 180.0) / (a / sqrtmagic * math.cos(radlat) * pi)
    gcj_lat = lat + dlat
    gcj_lng = lon + dlng
    return gcj_lng, gcj_lat

def get_node_coords_gcj():
    gcj_coords = {}
    for node, (lat_wgs, lon_wgs) in NODE_COORDS_WGS.items():
        lon_gcj, lat_gcj = wgs84_to_gcj02(lon_wgs, lat_wgs)
        gcj_coords[node] = (lat_gcj, lon_gcj)
    return gcj_coords

NODE_COORDS_GCJ = get_node_coords_gcj()

# 动态获取消防站
def get_current_stations():
    stations = []
    for name in ["清水塘街道消防站", "潮宗街政府专职消防救援站", "湘雅路街道文昌阁消防站"]:
        if name in POI_NODES:
            stations.append(FireStation(name, POI_NODES[name]))
    return stations

# 车辆配置
def create_vehicles_for_station():
    vehicles = []
    for i in range(1, 6):
        vehicles.append(Vehicle(f"S{i}", "small"))
    for i in range(1, 6):
        vehicles.append(Vehicle(f"M{i}", "medium"))
    for i in range(1, 4):
        vehicles.append(Vehicle(f"L{i}", "large"))
    return vehicles

if "station_vehicles" not in st.session_state:
    station_names = ["清水塘街道消防站", "潮宗街政府专职消防救援站", "湘雅路街道文昌阁消防站"]
    st.session_state.station_vehicles = {}
    for name in station_names:
        st.session_state.station_vehicles[name] = create_vehicles_for_station()

# 辅助函数
def select_best_station(fire_node, stations, scene, grid=None):
    if grid is None:
        grid = SpaceTimeGrid()
    best_station, best_time, best_path = None, float('inf'), []
    for station in stations:
        path, travel_time = dijkstra_weighted_path(G, station.position, fire_node, grid, scene)
        if path and travel_time < best_time:
            best_time, best_station, best_path = travel_time, station, path
    return best_station, best_time, best_path

def get_all_vehicles(stations):
    rows = []
    for station in stations:
        for v in st.session_state.station_vehicles[station.id]:
            rows.append({
                "车辆ID": v.id, "类型": v.type, "容量(吨)": v.capacity,
                "状态": "待命" if v.status=="idle" else "出警",
                "所属站点": station.id, "灌水时间(min)": v.fill_time
            })
    return rows

def get_total_capacity(stations):
    total = 0
    for station in stations:
        for v in st.session_state.station_vehicles[station.id]:
            if v.status == "idle":
                total += v.capacity
    return total

def calculate_water_supply(stations, level):
    return get_total_capacity(stations) >= WATER_DEMAND[level]

def compute_congestion_factor(dt, tp, et, ed, w):
    base = 1.0
    if dt == "节假日": base *= 1.2
    elif dt == "大型活动日": base *= 1.5
    if "高峰" in tp: base *= 1.8
    elif tp == "夜间": base *= 0.8
    if et != "无事件":
        if ed < 2: base *= 2.0
        elif ed < 5: base *= 1.5
        else: base *= 1.2
    if w == "暴雨": base *= 1.6
    elif w == "冰雪": base *= 2.0
    elif w == "雾霾": base *= 1.2
    return round(base, 2)

def get_path_coords_gcj(node_list):
    coords = []
    for node in node_list:
        coord = NODE_COORDS_GCJ.get(node)
        if coord:
            coords.append(coord)
    return coords

def ai_select_best_station(fire_node, stations, scene, grid, ai_models):
    fusion = ai_models['fusion']
    dqn = ai_models['dqn']
    fire_coord_wgs = NODE_COORDS_WGS.get(fire_node)
    if not fire_coord_wgs:
        return select_best_station(fire_node, stations, scene, grid)
    fire_lat_wgs, fire_lon_wgs = fire_coord_wgs

    # 获取实时多模态数据
    weather_data = fusion.get_weather_data(fire_lat_wgs, fire_lon_wgs)
    traffic_data = fusion.get_traffic_data(fire_lat_wgs, fire_lon_wgs)

    # 构建消防站状态
    stations_status = []
    for s in stations:
        s_coord_wgs = NODE_COORDS_WGS.get(s.position)
        dist_km = haversine(fire_lat_wgs, fire_lon_wgs, s_coord_wgs[0], s_coord_wgs[1])/1000 if s_coord_wgs else 5.0
        vehicles = st.session_state.station_vehicles.get(s.id, [])
        idle = [v for v in vehicles if v.status == 'idle']
        stations_status.append({
            'distance': dist_km,
            'idle_count': len(idle),
            'total_capacity': sum(v.capacity for v in idle),
            'avg_fill_time': sum(v.fill_time for v in idle)/max(1, len(idle)),
            'heavy_ratio': len([v for v in idle if v.type=='large'])/max(1, len(idle))
        })

    # 使用当前路况拥堵指数作为边权重因子
    congestion_factor = max(0.5, traffic_data['congestion_level'])  # 避免为0

    def ai_edge_time(u, v, scene, grid):
        base_len = G[u][v].get('length', 100.0)
        scene_multiplier = get_scene_weight_multiplier(scene, u, v, G)
        return base_len / SPEED * scene_multiplier * congestion_factor

    # DQN 选择消防站
    traffic_state = [traffic_data['congestion_level']]
    weather_encoding = fusion.get_weather_encoding(weather_data['weather'])
    state = dqn.get_state(st.session_state.fire_level, stations_status, traffic_state, weather_encoding)
    action_mask = [1] * len(stations)
    action = dqn.select_action(state, action_mask)
    selected_station = stations[action]

    # 计算路径
    try:
        path = nx.dijkstra_path(G, selected_station.position, fire_node,
                                weight=lambda u, v, d: ai_edge_time(u, v, scene, grid))
        total_time = nx.dijkstra_path_length(G, selected_station.position, fire_node,
                                             weight=lambda u, v, d: ai_edge_time(u, v, scene, grid))
    except nx.NetworkXNoPath:
        return None, float('inf'), []
    return selected_station, total_time, path

# 状态初始化
if "fire_level" not in st.session_state:
    st.session_state.fire_level = 2
if "scene" not in st.session_state:
    st.session_state.scene = "正常"
if "best_station_id" not in st.session_state:
    st.session_state.best_station_id = None
if "best_time" not in st.session_state:
    st.session_state.best_time = None
if "best_path" not in st.session_state:
    st.session_state.best_path = None
if "relay_result" not in st.session_state:
    st.session_state.relay_result = None
if "manual_path" not in st.session_state:
    st.session_state.manual_path = None
if "manual_time" not in st.session_state:
    st.session_state.manual_time = None
if "manual_station" not in st.session_state:
    st.session_state.manual_station = None
if "ai_models" not in st.session_state:
    st.session_state.ai_models = None

# 侧边栏
st.sidebar.title("动态因子配置")
scene = st.sidebar.selectbox("动态场景", ["正常", "演唱会", "体育赛事", "集市", "早高峰", "晚高峰", "夜间", "暴雨", "冰雪", "临时封路"])
if scene != st.session_state.scene:
    st.session_state.scene = scene
    st.session_state.best_station_id = None
    st.session_state.relay_result = None

date_type = st.sidebar.selectbox("日期类型", ["工作日", "周末", "节假日", "大型活动日"])
time_period = st.sidebar.selectbox("时段", ["早高峰 (7-9点)", "晚高峰 (17-19点)", "平峰", "夜间"])
event_type = st.sidebar.selectbox("周边事件类型", ["无事件", "体育赛事", "演唱会", "庙会集会", "临时管制"])
event_distance = st.sidebar.slider("事件距离 (km)", 0.0, 10.0, 3.0, 0.5)
weather = st.sidebar.selectbox("天气状况", ["晴朗", "暴雨", "冰雪", "雾霾"])
congestion_factor = compute_congestion_factor(date_type, time_period, event_type, event_distance, weather)
st.sidebar.metric("综合拥堵系数", congestion_factor)

st.sidebar.markdown("---")
st.sidebar.subheader("🤖 AI 智能增强模块")
use_ai = st.sidebar.checkbox("启用 AI 路况预测与调度", value=False)
if use_ai and st.session_state.ai_models is None:
    amap_key = st.sidebar.text_input("高德地图 API Key (可选)", type="password")
    weather_key = st.sidebar.text_input("天气 API Key (可选)", type="password")
    if st.sidebar.button("初始化 AI 模型"):
        with st.spinner("正在加载 AI 模型..."):
            num_nodes = G.number_of_nodes()
            dqn = DQNScheduler(state_dim=18, action_dim=20, device='cpu')
            fusion = MultiModalDataFusion(amap_key, weather_key, use_mock=(not amap_key))
            st.session_state.ai_models = { 'dqn': dqn, 'fusion': fusion}
            st.sidebar.success("AI 模型就绪")

st.sidebar.markdown("---")
st.sidebar.subheader("当前节点映射")
current_stations = get_current_stations()
for s in current_stations:
    st.sidebar.write(f"{s.id}: 节点 {s.position}")
st.sidebar.write(f"火场: 节点 {FIRE_NODE}")

if st.sidebar.button("重新计算最优调度"):
    with st.spinner(f"正在计算路径（场景: {st.session_state.scene}）..."):
        stations = get_current_stations()
        grid = SpaceTimeGrid()
        if use_ai and st.session_state.ai_models is not None:
            station, time_min, path = ai_select_best_station(FIRE_NODE, stations, st.session_state.scene, grid, st.session_state.ai_models)
        else:
            station, time_min, path = select_best_station(FIRE_NODE, stations, st.session_state.scene, grid)
        if station is None:
            st.sidebar.error("无法到达火场")
        else:
            st.session_state.best_station_id = station.id
            st.session_state.best_time = time_min
            st.session_state.best_path = path
            st.session_state.relay_result = None
            st.sidebar.success("调度完成！")
            if path:
                st.sidebar.write(f"路径起点节点: {path[0]}, 消防站节点: {station.position}")

# 大火时多车调度
if st.session_state.fire_level >= 3 and st.session_state.best_station_id and st.session_state.relay_result is None:
    with st.spinner("正在计算多车循环运输方案..."):
        fire_event = FireEvent("F1", FIRE_NODE, st.session_state.fire_level)
        water_source = WaterSource("供水点", WATER_SOURCE_NODE)
        stations = get_current_stations()
        for station in stations:
            station.vehicles = st.session_state.station_vehicles[station.id]
        result = schedule_relay(G, fire_event, stations, water_source, st.session_state.scene)
        if result[0] is not None:
            total_time, water_supplied, log, grid = result
            st.session_state.relay_result = {
                'total_time': total_time,
                'water_supplied': water_supplied,
                'log': log,
                'grid': grid
            }
        else:
            st.sidebar.error("多车调度失败，无可用车辆")

# 主界面
st.title("特种车辆动态路径优化与评估平台")
st.markdown("##### ——面向消防应急的智能调度系统")
st.markdown("---")

tab1, tab3 = st.tabs(["实时监控与调度", "参数配置与管理"])

with tab1:
    col_left, col_right = st.columns([0.35, 0.65], gap="medium")
    with col_left:
        st.subheader("多车状态面板")
        display_stations = get_current_stations()
        for station in display_stations:
            station.vehicles = st.session_state.station_vehicles[station.id]
        vehicle_df = pd.DataFrame(get_all_vehicles(display_stations))
        st.dataframe(vehicle_df, use_container_width=True, hide_index=True)

        st.markdown("---")
        st.subheader("火势等级")
        fire_level = st.radio(
            "火势等级", [1,2,3],
            format_func=lambda x: f"{x}级 {['小火(5t)','中火(35t)','大火(70t)'][x-1]}",
            index=st.session_state.fire_level-1, horizontal=True
        )
        if fire_level != st.session_state.fire_level:
            st.session_state.fire_level = fire_level
            st.session_state.best_station_id = None
            st.session_state.relay_result = None
            st.rerun()

        st.markdown("---")
        st.subheader("预警信息")
        if st.session_state.scene in ["演唱会","体育赛事","集市"]:
            st.warning(f"当前场景为「{st.session_state.scene}」，部分路段拥堵")
        elif st.session_state.scene in ["暴雨","冰雪"]:
            st.error(f"当前场景为「{st.session_state.scene}」，通行条件差")
        elif st.session_state.scene == "临时封路":
            st.error("临时封路，路径已自动避开")
        else:
            st.info(f"当前场景「{st.session_state.scene}」，道路通畅")

        st.markdown("---")
        st.subheader("方案推荐")
        water_ok = calculate_water_supply(display_stations, st.session_state.fire_level)
        if st.session_state.fire_level <= 2:
            st.success("推荐方案：直接派车（单次供水）")
            if not water_ok:
                st.warning("容量不足，建议增派车辆")
        else:
            st.info("推荐方案：循环运输（火场↔独立供水点接力）")
            if not water_ok:
                st.error("容量不足，无法满足大火需水量")
            elif st.session_state.relay_result:
                st.success(f"循环运输总时间：{st.session_state.relay_result['total_time']:.1f} 分钟")
                st.caption(f"总供水量：{st.session_state.relay_result['water_supplied']:.0f} 吨")
                fire_osm = OSM_COORDS["盛世荷园小区"]
                water_osm = OSM_COORDS["独立供水点"]
                dist_m = haversine(fire_osm[0], fire_osm[1], water_osm[0], water_osm[1])
                st.caption(f"供水点距离火场约 {dist_m:.0f} 米")

        if st.session_state.best_station_id:
            best_station_obj = next((s for s in display_stations if s.id == st.session_state.best_station_id), None)
            if best_station_obj:
                st.caption(f"最优消防站：**{best_station_obj.id}**")
            st.caption(f"预计到场时间：**{st.session_state.best_time:.1f} 分钟**")
            st.caption(f"当前场景：**{st.session_state.scene}**")
        else:
            st.caption("点击侧边栏「重新计算最优调度」查看推荐消防站")

    with col_right:
        st.subheader("实时位置与路径规划图")

        # 高德底图
        m = folium.Map(
            location=FIRE_GCJ_LOC,
            zoom_start=15,
            tiles='https://webrd0{s}.is.autonavi.com/appmaptile?lang=zh_cn&size=1&scale=1&style=8&x={x}&y={y}&z={z}',
            attr='高德地图',
            subdomains='1234',
            control_scale=True
        )

        folium.TileLayer(
            tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=6&x={x}&y={y}&z={z}',
            attr='高德卫星图',
            name='卫星图',
            subdomains='1234',
            overlay=False,
            control=True
        ).add_to(m)

        folium.TileLayer(
            tiles='https://webst0{s}.is.autonavi.com/appmaptile?style=7&x={x}&y={y}&z={z}',
            attr='高德标注',
            name='POI标注',
            subdomains='1234',
            overlay=True,
            control=True
        ).add_to(m)

        folium.LayerControl().add_to(m)

        # 绘制路网
        for u, v in G.edges():
            cu = NODE_COORDS_GCJ.get(u)
            cv = NODE_COORDS_GCJ.get(v)
            if cu and cv:
                folium.PolyLine([cu, cv], color="#888888", weight=2.5, opacity=0.7, tooltip="道路").add_to(m)

        # 消防站
        for station in display_stations:
            gcj = GCJ_COORDS.get(station.id)
            if gcj:
                folium.CircleMarker(
                    (gcj[1], gcj[0]), radius=12,
                    color="#FF4500", fill=True, fillColor="#FF4500", fillOpacity=0.9,
                    popup=station.id, tooltip=station.id
                ).add_to(m)

        # 火场
        fire_gcj = GCJ_COORDS["盛世荷园小区"]
        folium.CircleMarker(
            (fire_gcj[1], fire_gcj[0]), radius=12+4*st.session_state.fire_level,
            color="#DC143C", fill=True, fillColor="#DC143C", fillOpacity=0.9,
            popup="火场", tooltip="火场"
        ).add_to(m)
        folium.Marker(
            (fire_gcj[1], fire_gcj[0]),
            popup='盛世荷园小区火场',
            tooltip='火场位置',
            icon=folium.Icon(color='red', icon='fire', prefix='fa')
        ).add_to(m)

        # 供水点
        if st.session_state.fire_level >= 3 and WATER_SOURCE_NODE is not None:
            water_gcj = GCJ_COORDS["独立供水点"]
            folium.CircleMarker(
                (water_gcj[1], water_gcj[0]), radius=8, color="#00FF00", fill=True,
                fillColor="#00FF00", fillOpacity=0.7, popup="供水点", tooltip="供水点"
            ).add_to(m)
            folium.Marker(
                (water_gcj[1], water_gcj[0]),
                popup='独立供水点',
                tooltip='取水点',
                icon=folium.Icon(color='green', icon='tint', prefix='fa')
            ).add_to(m)

        # 地标
        hospital_gcj = GCJ_COORDS["湘雅医院"]
        folium.CircleMarker(
            (hospital_gcj[1], hospital_gcj[0]), radius=6, color="#9B59B6", fill=True,
            fillColor="#9B59B6", fillOpacity=0.6, popup="湘雅医院", tooltip="湘雅医院"
        ).add_to(m)
        sports_gcj = GCJ_COORDS["湖南省体育场"]
        folium.CircleMarker(
            (sports_gcj[1], sports_gcj[0]), radius=6, color="#9B59B6", fill=True,
            fillColor="#9B59B6", fillOpacity=0.6, popup="湖南省体育场", tooltip="湖南省体育场"
        ).add_to(m)

        # 最优路径
        if st.session_state.best_path:
            coords = get_path_coords_gcj(st.session_state.best_path)
            if len(coords) >= 2:
                folium.PolyLine(
                    coords, color="#FFD700", weight=5, opacity=0.9,
                    popup=f"最优路径（{st.session_state.scene}场景，{st.session_state.best_time:.1f}分钟）",
                    tooltip="最优路径"
                ).add_to(m)

        # 手动路径
        if st.session_state.manual_path:
            coords = get_path_coords_gcj(st.session_state.manual_path)
            if len(coords) >= 2:
                folium.PolyLine(
                    coords, color="#00FFFF", weight=4, opacity=0.8,
                    popup=f"{st.session_state.manual_station} → 火场（{st.session_state.manual_time:.1f}分钟）",
                    tooltip="手动选择路径"
                ).add_to(m)

        # 多车路径
        if st.session_state.fire_level >= 3 and st.session_state.relay_result and "log" in st.session_state.relay_result:
            vehicle_paths = {}
            for entry in st.session_state.relay_result['log']:
                if 'nodes' in entry and 'vehicle' in entry:
                    vid = entry['vehicle']; nodes = entry['nodes']
                    vehicle_paths.setdefault(vid, []).append(nodes)
            colors = ["#FF0000","#00FF00","#0000FF","#FFA500","#FF00FF","#00FFFF"]
            for idx, (vid, segs) in enumerate(vehicle_paths.items()):
                full_nodes = []
                for u, v in segs:
                    if not full_nodes: full_nodes = [u, v]
                    else: full_nodes.append(v if full_nodes[-1]==u else u)
                coords = get_path_coords_gcj(full_nodes)
                if len(coords) >= 2:
                    folium.PolyLine(
                        coords, color=colors[idx%len(colors)], weight=3, opacity=0.7,
                        popup=f"车辆 {vid} 路径", tooltip=f"车辆{vid}"
                    ).add_to(m)

        # 图例
        legend_html = """
        <div style="position: fixed; bottom: 30px; right: 30px; z-index: 1000; background: rgba(0,0,0,0.85); padding: 15px; border-radius: 15px; border: 2px solid #FF6B6B; color: white; font-family: 'Microsoft YaHei', sans-serif;">
            <h4 style="margin:0 0 10px 0; color:#FF6B6B;">图例说明</h4>
            <p><span style="background:#FF4500; width:20px; height:20px; display:inline-block; border-radius:50%;"></span> 消防站</p>
            <p><span style="background:#DC143C; width:20px; height:20px; display:inline-block; border-radius:50%;"></span> 火场</p>
            <p><span style="background:#00FF00; width:20px; height:20px; display:inline-block; border-radius:50%;"></span> 供水点（大火时）</p>
            <p><span style="background:#FFD700; width:20px; height:20px; display:inline-block;"></span> 最优路径</p>
            <p><span style="background:#00FFFF; width:20px; height:20px; display:inline-block;"></span> 手动选择路径</p>
            <p><span style="background:#888888; width:20px; height:20px; display:inline-block;"></span> 道路路网</p>
            <p><span style="background:#9B59B6; width:20px; height:20px; display:inline-block; border-radius:50%;"></span> 医院/体育场</p>
            <hr style="margin:8px 0;">
            <p style="font-size:12px;">🗺️ 底图: 高德地图 (GCJ02)</p>
            <p style="font-size:12px;">🚒 基于OSM路网 + 消防应急调度</p>
        </div>
        """
        m.get_root().html.add_child(folium.Element(legend_html))
        st_folium(m, width=None, height=600, key="fire_map")
        st.markdown("---")
        st.subheader("手动查看特定消防站路径")
        manual_station_name = st.selectbox(
            "选择消防站",
            [s.id for s in display_stations],
            key="manual_station_select"
        )
        if st.button("显示该消防站路径"):
            selected_station_node = next(s.position for s in display_stations if s.id == manual_station_name)
            with st.spinner("计算路径中..."):
                path, time = dijkstra_weighted_path(G, selected_station_node, FIRE_NODE, SpaceTimeGrid(), st.session_state.scene)
                if path:
                    st.session_state.manual_path = path
                    st.session_state.manual_time = time
                    st.session_state.manual_station = manual_station_name
                    st.success(f"路径计算完成，预计 {time:.1f} 分钟")
                else:
                    st.error("无法到达火场")
                    st.session_state.manual_path = None

        st.markdown("---")
        st.subheader("规划路径摘要")
        if st.session_state.best_path:
            st.caption(f"最优路径包含 {len(st.session_state.best_path)} 个节点")
            st.caption(f"预计时间：{st.session_state.best_time:.1f} 分钟")
        else:
            st.caption("尚未计算")

    st.markdown("---")
    col1, col2, col3, col4 = st.columns(4)
    with col1: st.metric("最优消防站到场时间", f"{st.session_state.best_time:.1f} min" if st.session_state.best_time else "未计算")
    with col2: supply_time = (st.session_state.best_time if st.session_state.fire_level <= 2 else (st.session_state.relay_result['total_time'] if st.session_state.relay_result else 0)); st.metric("供水链路总时间", f"{supply_time:.1f} min" if supply_time else "未计算")
    with col3: water_ok = calculate_water_supply(display_stations, st.session_state.fire_level); st.metric("供水能力", "充足" if water_ok else "可能不足")
    with col4: st.metric("当前场景", st.session_state.scene)
    total_cap = get_total_capacity(display_stations)
    st.caption(f"火灾等级 {st.session_state.fire_level} 总需水量：{WATER_DEMAND[st.session_state.fire_level]} 吨 | 当前车队总容量：{total_cap} 吨 | 调度策略：{'直接派车' if st.session_state.fire_level <= 2 else '循环运输'} | 动态场景：{st.session_state.scene}")

with tab3:
    st.subheader("系统参数配置")
    with st.expander("消防站与车辆资源管理", expanded=True):
        for station in display_stations:
            st.write(f"**{station.id}** (节点: {station.position})")
            df = pd.DataFrame([{"ID":v.id, "类型":v.type, "容量(吨)":v.capacity, "灌水时间(min)":v.fill_time, "状态":v.status} for v in st.session_state.station_vehicles[station.id]])
            st.dataframe(df, use_container_width=True, hide_index=True)
        if st.button("重置车辆配置"):
            station_names = ["清水塘街道消防站", "潮宗街政府专职消防救援站", "湘雅路街道文昌阁消防站"]
            for name in station_names:
                st.session_state.station_vehicles[name] = create_vehicles_for_station()
            st.session_state.best_station_id = None
            st.session_state.relay_result = None
            st.rerun()
    with st.expander("路网信息", expanded=True):
        st.markdown(f"**节点数量**：{G.number_of_nodes()}")
        st.markdown(f"**道路数量**：{G.number_of_edges()}")
        st.markdown("**数据来源**：")
        st.markdown("- 路网数据：OpenStreetMap (OSM)")
        st.markdown("- 底图样式：高德地图瓦片 (GCJ02)")
        st.markdown("- 坐标系统：路网 WGS84 → 显示 GCJ02（仅用于地图对齐）")
        st.markdown("**路网范围**：以盛世荷园小区为中心，半径2公里")