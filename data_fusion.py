import requests
import json
from datetime import datetime
import math

class MultiModalDataFusion:
    def __init__(self, amap_key=None, weather_key=None, use_mock=True):
        self.amap_key = amap_key
        self.weather_key = weather_key
        self.use_mock = use_mock or (amap_key is None or weather_key is None)

    def get_weather_data(self, lat, lon):
        if self.use_mock or not self.weather_key:
            hour = datetime.now().hour
            if 7 <= hour <= 9 or 17 <= hour <= 19:
                weather = '多云'; temp = 18
            elif 22 <= hour or hour <= 5:
                weather = '晴'; temp = 12
            else:
                weather = '晴'; temp = 22
            return {'temperature': temp, 'humidity': 60, 'wind_speed': 3.0, 'wind_direction': 180, 'weather': weather, 'rainfall': 0}
        else:
            url = "https://api.openweathermap.org/data/2.5/weather"
            params = {'lat': lat, 'lon': lon, 'appid': self.weather_key, 'units': 'metric'}
            try:
                resp = requests.get(url, params=params, timeout=5)
                data = resp.json()
                return {'temperature': data['main']['temp'], 'humidity': data['main']['humidity'], 'wind_speed': data['wind']['speed'], 'wind_direction': data['wind']['deg'], 'weather': data['weather'][0]['main'], 'rainfall': data.get('rain', {}).get('1h', 0)}
            except:
                return self.get_weather_data(lat, lon)

    def get_traffic_data(self, lat, lon, radius=1000):
        if self.use_mock or not self.amap_key:
            import random
            congestion = random.uniform(0.2, 0.8)
            return {'congestion_level': congestion, 'avg_speed': 30 * (1 - congestion)}
        else:
            delta = radius / 111000
            rectangle = f"{lon-delta},{lat-delta};{lon+delta},{lat+delta}"
            url = "https://restapi.amap.com/v3/traffic/status/rectangle"
            params = {'rectangle': rectangle, 'key': self.amap_key}
            try:
                resp = requests.get(url, params=params, timeout=5)
                data = resp.json()
                roads = data.get('trafficinfo', {}).get('roads', [])
                if not roads:
                    return {'congestion_level': 0.5, 'avg_speed': 30}
                total_speed = sum(road.get('speed', 60) for road in roads)
                avg_speed = total_speed / len(roads)
                congestion = 1 - (avg_speed / 60)
                congestion = max(0, min(1, congestion))
                return {'congestion_level': congestion, 'avg_speed': avg_speed}
            except:
                return {'congestion_level': 0.5, 'avg_speed': 30}

    def get_social_events(self, lat, lon, radius=2000):
        events = []
        sports_lat, sports_lon = 28.206789, 112.989565
        if self._haversine(lat, lon, sports_lat, sports_lon) < radius:
            events.append({'type': '体育赛事', 'name': '湖南湘涛主场赛事', 'time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'radius': 1500, 'impact_factor': 1.5})
        hospital_lat, hospital_lon = 28.21224, 112.984105
        if self._haversine(lat, lon, hospital_lat, hospital_lon) < radius:
            events.append({'type': '医疗聚集', 'name': '湘雅医院就诊高峰', 'time': datetime.now().strftime('%Y-%m-%d %H:%M'), 'radius': 800, 'impact_factor': 1.3})
        return events

    def _haversine(self, lat1, lon1, lat2, lon2):
        R = 6371000
        phi1 = math.radians(lat1); phi2 = math.radians(lat2)
        dphi = math.radians(lat2 - lat1); dlambda = math.radians(lon2 - lon1)
        a = math.sin(dphi/2)**2 + math.cos(phi1)*math.cos(phi2)*math.sin(dlambda/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        return R * c

    def get_weather_encoding(self, weather_str):
        mapping = {'晴朗':0.0, '晴':0.0, '多云':0.2, '阴':0.3, '小雨':0.4, '中雨':0.6, '大雨':0.8, '暴雨':1.0, '雪':0.7, '雾':0.5}
        for key, val in mapping.items():
            if key in weather_str:
                return val
        return 0.2