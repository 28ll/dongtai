import networkx as nx
import heapq
from typing import List
from data_models import FireStation, WaterSource, FireEvent, Vehicle, WATER_DEMAND, DISCHARGE_RATE
from space_time import SpaceTimeGrid
from one_vehicle import dijkstra_weighted_path, get_edge_time

def travel_path(path, start_time, G, scene, grid, vehicle_id):
    time = start_time
    occupancies = []
    for i in range(len(path)-1):
        u, v = path[i], path[i+1]
        t = get_edge_time(G, u, v, scene, grid)
        time = grid.get_next_free_time((u, v), time, t)
        grid.reserve((u, v), time, time+t)
        occupancies.append({
            'edge': (u, v),
            'nodes': [u, v],
            'start': time,
            'end': time+t,
            'vehicle': vehicle_id
        })
        time += t
    return occupancies, time

class VehicleState:
    def __init__(self, vehicle: Vehicle, station: FireStation):
        self.vehicle = vehicle
        self.current_pos = station.position
        self.time = 0.0
        self.remaining_capacity = vehicle.capacity

def schedule_relay(G, fire_event: FireEvent, stations: List[FireStation],
                   water_source: WaterSource, scene: str):
    grid = SpaceTimeGrid()
    vehicles_with_path = []
    for station in stations:
        for v in station.vehicles:
            if v.status == 'idle':
                path, travel_time = dijkstra_weighted_path(G, station.position, fire_event.position, grid, scene)
                if path:
                    vehicles_with_path.append((station, v, path, travel_time))
    if not vehicles_with_path:
        return None, None, None, None

    needed = WATER_DEMAND[fire_event.level]
    vehicles_with_path.sort(key=lambda x: x[1].capacity, reverse=True)
    station_to_vehicles = {}
    for station, v, path, travel_time in vehicles_with_path:
        station_to_vehicles.setdefault(station, []).append((v, path, travel_time))
    candidates = []
    for station, veh_list in station_to_vehicles.items():
        best = max(veh_list, key=lambda x: x[0].capacity)
        candidates.append((station, best[0], best[1], best[2]))
    candidates.sort(key=lambda x: x[1].capacity, reverse=True)
    selected = []
    total_capacity = 0
    used_stations = set()
    for station, v, path, travel_time in candidates:
        if total_capacity >= needed: break
        selected.append((station, v, path, travel_time))
        total_capacity += v.capacity
        used_stations.add(station)
    if total_capacity < needed:
        remaining_needed = needed - total_capacity
        selected_keys = {(s.id, v.id) for s, v, _, _ in selected}
        all_remaining = [(station, v, path, travel_time) for station, v, path, travel_time in vehicles_with_path
                         if (station.id, v.id) not in selected_keys]
        all_remaining.sort(key=lambda x: x[1].capacity, reverse=True)
        for station, v, path, travel_time in all_remaining:
            if total_capacity >= needed: break
            selected.append((station, v, path, travel_time))
            total_capacity += v.capacity
            used_stations.add(station)
    vehicle_states = {}
    event_heap = []
    schedule_log = []
    for station, v, path, travel_time in selected:
        state = VehicleState(v, station)
        vehicle_states[v.id] = state
        occupancies, arrival_time = travel_path(path, 0, G, scene, grid, v.id)
        schedule_log.extend(occupancies)
        state.time = arrival_time
        state.current_pos = fire_event.position
        heapq.heappush(event_heap, (arrival_time, v.id, 'discharge', fire_event.position))
    total_water_supplied = 0
    while event_heap and total_water_supplied < needed:
        time, vid, task, pos = heapq.heappop(event_heap)
        state = vehicle_states[vid]
        vehicle = state.vehicle
        if task == 'discharge':
            discharge = min(vehicle.capacity, needed - total_water_supplied)
            discharge_time = discharge / DISCHARGE_RATE
            schedule_log.append({'time': time, 'vehicle': vid, 'action': f'discharge {discharge}t', 'duration': discharge_time})
            total_water_supplied += discharge
            state.remaining_capacity = vehicle.capacity - discharge
            finish_time = time + discharge_time
            state.time = finish_time
            path, _ = dijkstra_weighted_path(G, fire_event.position, water_source.position, grid, scene)
            if not path: continue
            occupancies, arrival_water = travel_path(path, finish_time, G, scene, grid, vid)
            schedule_log.extend(occupancies)
            state.time = arrival_water
            state.current_pos = water_source.position
            heapq.heappush(event_heap, (arrival_water, vid, 'refill', water_source.position))
        elif task == 'refill':
            fill_time = vehicle.fill_time
            schedule_log.append({'time': time, 'vehicle': vid, 'action': 'refill', 'duration': fill_time})
            finish_time = time + fill_time
            state.time = finish_time
            state.remaining_capacity = vehicle.capacity
            path, _ = dijkstra_weighted_path(G, water_source.position, fire_event.position, grid, scene)
            if not path: continue
            occupancies, arrival_fire = travel_path(path, finish_time, G, scene, grid, vid)
            schedule_log.extend(occupancies)
            state.time = arrival_fire
            state.current_pos = fire_event.position
            heapq.heappush(event_heap, (arrival_fire, vid, 'discharge', fire_event.position))
    if vehicle_states:
        total_time = max(state.time for state in vehicle_states.values())
    else:
        total_time = 0
    return total_time, total_water_supplied, schedule_log, grid