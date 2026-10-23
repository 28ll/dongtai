import networkx as nx
from space_time import SpaceTimeGrid
from road_network import get_scene_weight_multiplier, SPEED

def get_edge_time(G, u, v, scene, grid):
    try:
        base_len = G[u][v].get('length', 100.0)
    except (KeyError, AttributeError):
        base_len = 100.0
    if base_len <= 0:
        base_len = 100.0

    congestion = grid.get_congestion((u, v))
    if not isinstance(congestion, (int, float)):
        congestion = 1.0
    scene_multiplier = get_scene_weight_multiplier(scene, u, v, G)
    if not isinstance(scene_multiplier, (int, float)):
        scene_multiplier = 1.0

    return base_len / SPEED * scene_multiplier * congestion

def dijkstra_weighted_path(G, start, end, grid, scene):
    try:
        path = nx.dijkstra_path(G, start, end, weight=lambda u, v, d: get_edge_time(G, u, v, scene, grid))
        total_time = nx.dijkstra_path_length(G, start, end, weight=lambda u, v, d: get_edge_time(G, u, v, scene, grid))
        return path, total_time
    except nx.NetworkXNoPath:
        print(f"警告：从 {start} 到 {end} 无路径")
        return [], float('inf')
    except Exception as e:
        print(f"错误：计算路径时异常 {e}，起点 {start}，终点 {end}")
        return [], float('inf')