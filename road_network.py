import networkx as nx
import math
import os
from typing import Dict, Tuple, Optional

SPEED =600

def haversine(lat1, lon1, lat2, lon2):
    try:
        lat1 = float(lat1); lon1 = float(lon1); lat2 = float(lat2); lon2 = float(lon2)
    except (TypeError, ValueError):
        return float('inf')
    R = 6371000
    phi1 = math.radians(lat1); phi2 = math.radians(lat2)
    dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

# ----------  OSM 坐标 (WGS84) ----------
OSM_COORDS = {
    "盛世荷园小区": (28.210667, 112.977931),
    "清水塘街道消防站": (28.205934, 112.984220),
    "潮宗街政府专职消防救援站": (28.206136, 112.968982),
    "湘雅路街道文昌阁消防站": (28.219450, 112.974300),
    "湘雅医院": (28.215490, 112.978270),
    "湖南省体育场": (28.210080, 112.983910),
    "独立供水点": (28.210630, 112.973220)
}

# ---------- 高德坐标 (GCJ02) ----------
GCJ_COORDS = {
    "盛世荷园小区": (112.983063, 28.207196),
    "清水塘街道消防站": (112.988777,28.201324),
    "潮宗街政府专职消防救援站": (112.974695, 28.202916),
    "湘雅路街道文昌阁消防站": (112.980035, 28.215614),
    "湘雅医院": (112.984105, 28.21224),
    "湖南省体育场": (112.989565, 28.206789),
    "独立供水点": (112.979001,28.207054)
}

def build_road_network(graphml_path: str = "road_network.graphml") -> nx.Graph:
    if not os.path.exists(graphml_path):
        raise FileNotFoundError(f"路网文件 {graphml_path} 不存在。")
    G = nx.read_graphml(graphml_path)
    for node, data in G.nodes(data=True):
        if 'y' in data and 'x' in data:
            G.nodes[node]['pos'] = (data['y'], data['x'])
        else:
            G.nodes[node]['pos'] = None
    print(f"路网加载完成：节点数 {G.number_of_nodes()}, 边数 {G.number_of_edges()}")
    return G

def get_node_coords(G: nx.Graph) -> Dict[int, Tuple[float, float]]:
    valid = {}
    for node, pos in G.nodes(data='pos'):
        if pos is None: continue
        if isinstance(pos, (tuple, list)) and len(pos) == 2:
            try:
                lat, lon = float(pos[0]), float(pos[1])
                valid[node] = (lat, lon)
            except: continue
    return valid

def find_nearest_node(G: nx.Graph, lat: float, lon: float) -> Optional[int]:
    best_node = None
    best_dist = float('inf')
    for node, pos in G.nodes(data='pos'):
        if pos is None: continue
        if not isinstance(pos, (tuple, list)) or len(pos) != 2: continue
        try:
            node_lat, node_lon = float(pos[0]), float(pos[1])
            dist = haversine(lat, lon, node_lat, node_lon)
        except: continue
        if dist < best_dist:
            best_dist = dist
            best_node = node
    if best_node is not None:
        print(f"  最近节点 {best_node} 距离 {best_dist:.1f} 米")
    else:
        print(f"  警告：未找到任何有效节点")
    return best_node

def get_scene_weight_multiplier(scene: str, u: int, v: int, G: nx.Graph) -> float:
    edge_data = G.get_edge_data(u, v)
    if not edge_data: return 1.0
    if isinstance(edge_data, dict):
        first_edge = list(edge_data.values())[0]
    else:
        first_edge = edge_data
    highway = first_edge.get('highway', 'unknown')

    def get_node_coord(node):
        pos = G.nodes[node].get('pos')
        if pos and isinstance(pos, (tuple, list)) and len(pos) == 2:
            try: return float(pos[0]), float(pos[1])
            except: return None
        return None

    # 使用 OSM 坐标判断附近
    hospital_osm = OSM_COORDS["湘雅医院"]
    sports_osm = OSM_COORDS["湖南省体育场"]
    is_near_hospital = False
    for node in [u, v]:
        coord = get_node_coord(node)
        if coord:
            try:
                if haversine(coord[0], coord[1], hospital_osm[0], hospital_osm[1]) < 200:
                    is_near_hospital = True
                    break
            except: continue

    is_near_sports = False
    for node in [u, v]:
        coord = get_node_coord(node)
        if coord:
            try:
                if haversine(coord[0], coord[1], sports_osm[0], sports_osm[1]) < 300:
                    is_near_sports = True
                    break
            except: continue

    if scene in ["早高峰", "晚高峰"]:
        if is_near_hospital: return 3.5
        elif highway in ['primary','secondary','trunk']: return 2.0
        elif highway in ['motorway','motorway_link']: return 2.5
        else: return 1.3
    elif scene == "演唱会":
        return 3.5 if is_near_sports else 1.2
    elif scene == "体育赛事":
        return 3.0 if is_near_sports else 1.1
    elif scene == "夜间":
        return 0.8
    elif scene == "暴雨":
        return 1.6
    elif scene == "冰雪":
        return 2.0
    elif scene == "临时封路":
        return 9999 if ("芙蓉路" in str(highway)) else 1.2
    else:
        return 1.0

def get_poi_nodes(G: nx.Graph) -> Dict[str, int]:
    poi_nodes = {}
    print("\n===== POI 节点映射 (OSM坐标) =====")
    for name, (lat, lon) in OSM_COORDS.items():
        node_id = find_nearest_node(G, lat, lon)
        if node_id is None:
            print(f"  错误：找不到最近节点，跳过 {name}")
            continue
        poi_nodes[name] = node_id
        pos = G.nodes[node_id].get('pos')
        if pos and len(pos) == 2:
            try:
                dist = haversine(lat, lon, float(pos[0]), float(pos[1]))
                print(f"{name} -> 节点 {node_id} (距离 {dist:.1f}米)")
            except: pass
    print("=========================\n")
    return poi_nodes

def get_gcj_coords():
    return GCJ_COORDS
# 在 road_network.py 末尾添加

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