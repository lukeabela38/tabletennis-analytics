import json, re, time
from prometheus_client import CollectorRegistry, Gauge, push_to_gateway
from prometheus_client.core import GaugeMetricFamily
from prometheus_client.registry import Collector
from tqdm import tqdm

class CustomGauge(Collector):
    def __init__(self, body_part, dict):
        self.body_part = body_part
        self.dict = dict

    def collect(self):

        body_part = self.clean_metric_name(self.body_part)
        g = GaugeMetricFamily(body_part, 'x, y, z, visibility, presence values', labels=[body_part])
        for k,v in self.dict.items():
            metric = self.clean_metric_name(k)
            g.add_metric([metric], v)
        yield g

    @staticmethod
    def clean_metric_name(s: str, pattern: str = "[^0-9a-zA-Z\s]+") -> str:
        s = re.sub(pattern, "", s)
        s = s.strip()
        s = s.replace(" ", "_")
        return f"data_{s}_value"
    
def main():

    for i in tqdm(range(1,230,1)):
        registry = CollectorRegistry()

        g = Gauge('job_last_success_unixtime', 'Last time a batch job successfully finished', registry=registry)
        g.set_to_current_time()

        with open(f"malong/livestream_{i}.json") as json_file:
            metrics = json.load(json_file)
    
        for key,value in metrics.items():
            registry.register(CustomGauge(key, value))

        time.sleep(1)
        push_to_gateway('localhost:9091', job='Pose Estimation', registry=registry)

    return 0

if __name__ == "__main__":
    main()
