#!/usr/bin/env python3

#%%
jobid='10931852'
nodeid='n065'
f = open(f"../log2/{jobid}/{nodeid}.out", "r")
lines = f.readlines()
world_size = 32
dp_dim = 4
num_iter = 10
df_list = []


import pandas as pd
import plotly.express as px
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import tqdm
items= ['forward-compute',
'post-process',
'loss-compute',
'backward-compute',
'batch-generator',
'forward-recv',
'forward-send',
'backward-recv',
'backward-send',
'forward-send-forward-recv',
'forward-send-backward-recv',
'backward-send-forward-recv',
'backward-send-backward-recv',
'forward-backward-send-forward-backward-recv',
'layernorm-grads-all-reduce',
'embedding-grads-all-reduce',
'grads-all-reduce',
'grads-reduce-scatter',
'params-all-gather',
'optimizer-copy-to-main-grad',
'optimizer-unscale-and-check-inf',
'optimizer-clip-main-grad',
'optimizer-count-zeros',
'optimizer-inner-step',
'optimizer-copy-main-to-model-params',
'optimizer'
       ]

colors=[
    "blue",
    "lime",
    "green",
    "red",
    "purple",
    "brown",
    "olive",
    "tan",
    "yellow",
    "pink",
    "black",
    "cadetblue",
    "chocolate",
    "cyan",
    "magenta",
    "gold",
    "darkgreen",
    "darkred",
    "azure",
    "khaki",
    "skyblue",
    "indigo",
    "lightgreen",
    "orange",
    "firebrick",
    "navy"
]

group4 = [
'forward-send-forward-recv',
'backward-send-backward-recv',
'forward-backward-send-forward-backward-recv',
'layernorm-grads-all-reduce',
'grads-reduce-scatter',
'params-all-gather',

] 

group3 = [
'forward-recv',
'backward-send',
'forward-send',
'backward-recv',
'grads-all-reduce',
'embedding-grads-all-reduce',
'backward-send-forward-recv',
'forward-send-backward-recv',
] 

group2 = [
'forward-compute',
'backward-compute',
'optimizer',
'layrnorm-grads-all-reduce'
] 

group1 = [
'batch-generator',
'optimizer-copy-to-main-grad',
'optimizer-copy-main-to-model-params',
'optimizer-unscale-and-check-inf',
'optimizer-clip-main-grad',
'optimizer-count-zeros',
'optimizer-inner-step',
'post-process',
'loss-compute',
] 
color_map = {}
    

for idx, color_name in enumerate(colors):
    css4 = mcolors.CSS4_COLORS[color_name]
    rgba = mcolors.to_rgba_array([css4])
    color_map[items[idx]] = list(rgba[0])

def parse_df(world_size, dp_dim, num_iter, lines, i):
    iteration = (i // 10) * 10
    iter_idx = i % 10
    data = []
    parse = False
    for idx, line in enumerate(lines):
        if f"iteration       {iteration}" in line:
            parse = True
        if parse:
            for item in items:
                if item+":" in line:
                    start_rank = int(lines[idx + 1].split(":")[0].split()[1])
                    for rank in range(world_size):
                        start_line = lines[idx + 2 + rank*3]
                        end_line = lines[idx + 3 + rank*3]
                        if (item in start_line) and (item in end_line):
                            start_list = list(map(float, start_line.split(":")[1].split()))
                            end_list = list(map(float,end_line.split(":")[1].split()))

                            counts = len(start_list) 
                            if counts < num_iter:
                                continue
                            data_item = {
                                    'rank': rank+start_rank,
                                    'task': item,
                                    'length': counts
                            }
                            for mini_idx in range(counts):
                                data_item[f'start{mini_idx}']= start_list[mini_idx]
                                data_item[f'end{mini_idx}'] = end_list[mini_idx]
                                data_item[f'delta{mini_idx}'] =  end_list[mini_idx]- start_list[mini_idx]
                            data.append(data_item)
        if f"iteration       {iteration+10}" in line:
            parse = False
            
    df = pd.DataFrame(data)
    plot_datas=[]
    outname=f"./figure{iter_idx}.html"
    plot_color_map = {}
    for idx, item in enumerate(items):
        itemdata = df.loc[df["task"] == item, :]
        for yid in range(len(itemdata)):
            row = itemdata.iloc[yid, :]
            for xid in range((row['length']//num_iter) * iter_idx, (row['length']//num_iter) * (iter_idx + 1)):
                rank = row['rank']-0.5 + idx/len(items)
                if item in group1:
                    rank = row['rank']-0.5 +  0   + 1/6
                if item in group2:
                    rank = row['rank']-0.5 + 1/3  + 1/6          
                if item in group3:
                    rank = row['rank']-0.5 + 2/3  + 1/6
                start = row[f'start{xid}']
                finish = row[f'end{xid}']
                delta = finish - start
                color = 'rgba'+str(tuple(color_map[item][:-1] + [((xid - (row["length"]//num_iter) * iter_idx) /(row['length']//num_iter) + 1/(row['length']//num_iter) ) * 0.8+0.2]))
                label=f'{item}-{xid - (row["length"]//num_iter) * iter_idx}'
                plot_datas.append(
                    {
                        "rank": rank,
                        "start": start,
                        "finish": finish,
                        "delta": delta,
                        "task":item,
                        "color": color,
                        "label": label}   
                )
                plot_color_map[label] = color

    plot_datas = pd.DataFrame(plot_datas)
    plot_datas['rank'] = plot_datas['rank'].astype(float)

    # minimun = plot_datas.start.min()
    # plot_datas.start = plot_datas.start-minimun
    # plot_datas.finish = plot_datas.finish-minimun
    plot_datas['task'] =plot_datas['task'].apply(lambda x: x.strip())
    return plot_datas, plot_color_map





def plot(plot_datas, plot_color_map):
    fig = px.timeline(plot_datas, 
                      x_start="start", 
                      x_end="finish", 
                      y="rank", 
                      color="label",
                      labels={
                         "label": "task",
                         "start": "start(s)",
                         "finish": "finish(s)"
                     },
                      color_discrete_map=plot_color_map,
                      width=1600, 
                      height=800
                     )

    fig.layout.xaxis.type = 'linear'
    for d in fig.data:
        filt = plot_datas['label'] == d.name
        d.x = plot_datas[filt]['delta'].tolist()
    for rank in range(world_size):
        fig.add_hline(y=rank-0.5, line_width=1, opacity=0.3)

    for dp_idx in range(world_size // dp_dim + 1):
        fig.add_hline(y=dp_idx*dp_dim-0.5, line_width=1)


    fig.update_yaxes(tickvals=[str(rank) for rank  in range(world_size)])
    fig.update_traces(width=1/3)
    fig.write_html("out.html")
    
    
for i in tqdm.tqdm(range(10, 50)):
    df, plot_color_map = parse_df(world_size,dp_dim,num_iter, lines,i)
    df_list.append(df)
total_df = pd.concat(df_list)
plot(total_df, plot_color_map)