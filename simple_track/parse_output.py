import datetime as dt

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

STORM_PARSE_MAP = {
    'area': int,
    'centroid': lambda x: [float(v) for v in x.split(',')],
    'box': lambda x: [int(v) for v in x.split(',')],
    'life': int,
    'dx': float,
    'dy': float,
    'meanv': float,
    'extreme': float,
    'accreted': lambda x: [int(v) for v in x.split(',')],
    'parent': int,
    'child': lambda x: [int(v) for v in x.split(',')],
}

MISSING_VALUE = -999


class Storm:
    def __init__(self, storm_dict):
        for k, v in storm_dict.items():
            setattr(self, k, v)


def parse_storms(tracking_output):
    storms = {}
    for out in tracking_output:
        print(out)
        lines = out.read_text().split('\n')

        missing_value = int(lines[0].split('=')[1])
        assert missing_value == MISSING_VALUE, 'wrong missing value!'
        start_date = dt.datetime.strptime(lines[1].split('=')[1], '%d/%m/%y-%H%M')
        curr_date = dt.datetime.strptime(lines[2].split('=')[1], '%d/%m/%y-%H%M')

        label_method = lines[3].split('=')[1]
        squarelength = float(lines[4].split('=')[1])
        rafraction = float(lines[5].split('=')[1])
        total_num_storms = int(lines[6].split('=')[1])

        storms_at_time = {}
        for line in [l for l in lines[7:] if l]:
            split_line = line.split()
            storm_dict = {'id': int(split_line[1])}
            storm_dict.update({k: STORM_PARSE_MAP[k](v) for k, v in [ll.split('=') for ll in split_line[2:]]})
            storms_at_time[storm_dict['id']] = Storm(storm_dict)
        storms[curr_date] = storms_at_time
    return storms


def create_storm_dag(storms):
    storm_dag = nx.DiGraph()
    for d1, d2 in zip(storms.keys(), list(storms.keys())[1:]):
        storms1 = storms[d1]
        storms2 = storms[d2]

        common_ids = set(storms1.keys()) & set(storms2.keys())
        for storm_id in sorted(common_ids):
            s1 = storms1[storm_id]
            s2 = storms2[storm_id]

            storm_dag.add_edge(s1, s2)
        for s2 in storms2.values():
            if s2.parent != MISSING_VALUE:
                p = storms1[s2.parent]
                storm_dag.add_edge(p, s2)
            for acc in [a for a in s2.accreted if a != MISSING_VALUE]:
                acc_storm = storms1[acc]
                storm_dag.add_edge(acc_storm, s2)
    return storm_dag


def plot_storms_dag(storms, storm_dag, display='dag'):
    pos = {}
    max_id = 0
    if display == 'dag':
        for i, (date, storms_at_time) in enumerate(storms.items()):
            for j, storm in enumerate(storms_at_time.values()):
                pos[storm] = np.array([i, storm.id * 5])
                max_id = max(max_id, storm.id)
    elif display == 'loc':
        for i, (date, storms_at_time) in enumerate(storms.items()):
            for j, storm in enumerate(storms_at_time.values()):
                pos[storm] = np.array(storm.centroid)

    plt.figure(display)
    plt.clf()
    node_size = 20
    nx.draw_networkx_nodes(
        storm_dag, pos, storm_dag.nodes, node_color=[s.id for s in storm_dag.nodes], node_size=node_size
    )
    if display == 'dag':
        plt.xlim((0, len(storms)))
        plt.ylim((0, max_id * 5))
    plt.pause(0.01)
    plt.savefig(f'output/storms_dag.{display}.png')
