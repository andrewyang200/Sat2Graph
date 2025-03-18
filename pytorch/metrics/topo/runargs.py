import os


tiles = [119]


def run_command(tile_id):
    for i in tile_id:
        command = f"python main.py -graph_gt /Users/andrewyang/PycharmProjects/Sat2Graph/Sat2Graph-Server/data/20cities/region_{i}_refine_gt_graph.p " \
                  f"-graph_prop /Users/andrewyang/PycharmProjects/pytorch/outputs/region_{i}_output_graph.p -output topo_region_{i}.txt"
        os.system(command)


run_command(tiles)
