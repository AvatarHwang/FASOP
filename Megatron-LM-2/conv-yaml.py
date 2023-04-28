#%%
import yaml
import pandas as pd
pd.set_option('display.max_rows', 1000)
import matplotlib.pyplot as plt
f = open("2216-ours",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
ours = pd.DataFrame(data)

f = open("2216-amp",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
amp = pd.DataFrame(data)

#%%
amp
# %%

ours.loc['rank 55':'rank 63',:]

# %%

amp.columns
# %%
# ours.mean()
print(ours.loc[:'rank  7',['forward-compute','backward-compute']].mean())
print(ours.loc['rank  7':'rank 15',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 15':'rank 23',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 23':'rank 31',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 31':'rank 39',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 39':'rank 47',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 47':'rank 55',['forward-compute','backward-compute']].mean())
print(ours.loc['rank 55':'rank 63',['forward-compute','backward-compute']].mean())
# %%
print(amp.loc[:'rank  7',['forward-compute','backward-compute']].mean())
print(amp.loc['rank  7':'rank 15',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 15':'rank 23',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 23':'rank 31',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 31':'rank 39',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 39':'rank 47',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 47':'rank 55',['forward-compute','backward-compute']].mean())
print(amp.loc['rank 55':'rank 63',['forward-compute','backward-compute']].mean())
# %%
amp.loc[:,['forward-compute','backward-compute']].mean()
# %%
ours.loc[:,['forward-compute','backward-compute']].mean()
# %%
print(ours.loc[:'rank  7','forward-backward'].mean())
print(ours.loc['rank  7':'rank 15','forward-backward'].mean())
print(ours.loc['rank 15':'rank 23','forward-backward'].mean())
print(ours.loc['rank 23':'rank 31','forward-backward'].mean())
print(ours.loc['rank 31':'rank 39','forward-backward'].mean())
print(ours.loc['rank 39':'rank 47','forward-backward'].mean())
print(ours.loc['rank 47':'rank 55','forward-backward'].mean())
print(ours.loc['rank 55':'rank 63','forward-backward'].mean())
# %%
print(amp.loc[:'rank  7','forward-backward'].mean())
print(amp.loc['rank  7':'rank 15','forward-backward'].mean())
print(amp.loc['rank 15':'rank 23','forward-backward'].mean())
print(amp.loc['rank 23':'rank 31','forward-backward'].mean())
print(amp.loc['rank 31':'rank 39','forward-backward'].mean())
print(amp.loc['rank 39':'rank 47','forward-backward'].mean())
print(amp.loc['rank 47':'rank 55','forward-backward'].mean())
print(amp.loc['rank 55':'rank 63','forward-backward'].mean())
# %%
print(ours.loc[:'rank  7','forward-send-backward-recv'].mean())
print(ours.loc['rank  7':'rank 15','forward-send-backward-recv'].mean())
print(ours.loc['rank 15':'rank 23','forward-send-backward-recv'].mean())
print(ours.loc['rank 23':'rank 31','forward-send-backward-recv'].mean())
print(ours.loc['rank 31':'rank 39','forward-send-backward-recv'].mean())
print(ours.loc['rank 39':'rank 47','forward-send-backward-recv'].mean())
print(ours.loc['rank 47':'rank 55','forward-send-backward-recv'].mean())
print(ours.loc['rank 55':'rank 63','forward-send-backward-recv'].mean())
# %%
print(amp.loc[:'rank  7','forward-send-backward-recv'].mean())
print(amp.loc['rank  7':'rank 15','forward-send-backward-recv'].mean())
print(amp.loc['rank 15':'rank 23','forward-send-backward-recv'].mean())
print(amp.loc['rank 23':'rank 31','forward-send-backward-recv'].mean())
print(amp.loc['rank 31':'rank 39','forward-send-backward-recv'].mean())
print(amp.loc['rank 39':'rank 47','forward-send-backward-recv'].mean())
print(amp.loc['rank 47':'rank 55','forward-send-backward-recv'].mean())
print(amp.loc['rank 55':'rank 63','forward-send-backward-recv'].mean())
# %%

# %%

for col in ours.columns:
    print(col)
    plt.plot(amp.loc[:,col].reset_index().iloc[:,-1])
    plt.plot(ours.loc[:,col].reset_index().iloc[:,-1])
    plt.legend(['amp','ours'])
    plt.title(col)
    plt.xlabel('[ gpu rank ]')
    plt.ylabel('[ elapsed time (ms) ]')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
    plt.ylim(0,amp.loc[:,col].reset_index().iloc[:,-1].max())
    for i in range(16):
        plt.axvline(i*4, color='k', linestyle='--')
    plt.show()


# %%

plt.plot((amp.loc[:,'backward-compute']+amp.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-o')
plt.plot((ours.loc[:,'backward-compute']+ours.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-x')
plt.legend(['Model Partition: '+'3-'*15+'3','Model Partition: 4'+('-3'*14)+'-2'])
plt.xlabel('GPU rank')
plt.ylabel('Per Iter. Time (ms)')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
for i in range(16):
    plt.axvline(i*4, color='k', linestyle='--')
plt.show()
plt.savefig('1248-computation.png', dpi=1200)

# %%
(amp.loc[:,'backward-compute']+amp.loc[:,'forward-compute']).reset_index().iloc[:,-1]
# %%
(ours.loc[:,'backward-compute']+ours.loc[:,'forward-compute']).reset_index().iloc[:,-1]

# %%
import pandas as pd

data = pd.read_csv('d.tsv', sep='	')
# %%
# %%
import matplotlib.pyplot as plt
# %%
data.loc[0,"gpus"]
# %%

plt.scatter(data['money'][:100], data['latency'][:100])
# %%
data["homo"] = data["gpus"].apply(lambda x: 1 if 0 in eval(x) else 0)
# %%
data.homo.value_counts()

# %%
plt.scatter(data.loc[data.homo==1, 'money'], 1 / data.loc[data.homo==1, 'latency'])
plt.scatter(data.loc[data.homo==0, 'money'], 1 / data.loc[data.homo==0, 'latency'])
plt.xlim(0,10000000)
# %%



import yaml
import pandas as pd
pd.set_option('display.max_rows', 1000)
import matplotlib.pyplot as plt
f = open("2216-ours",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
ours = pd.DataFrame(data)

f = open("2216-amp",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
amp = pd.DataFrame(data)
plt.plot((amp.loc[:,'backward-compute']+amp.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-o')
plt.plot((ours.loc[:,'backward-compute']+ours.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-x')
plt.legend(['AMP '+'3-'*15+'3','FASOP 4'+('-3'*14)+'-2'])
plt.xlabel('GPU rank')
plt.ylabel('Per Iter. Time (ms)')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
for i in range(16):
    plt.axvline(i*4, color='k', linestyle='--')

plt.tight_layout()
plt.savefig('12216-computation.png')

# %%
ours.loc[:,'grads-all-reduce']
amp.loc[:,'grads-all-reduce']


#%%
import yaml
import pandas as pd
pd.set_option('display.max_rows', 1000)
import matplotlib.pyplot as plt
f = open("248-ours",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
ours = pd.DataFrame(data)

f = open("248-amp",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
amp = pd.DataFrame(data)
plt.plot((amp.loc[:,'backward-compute']+amp.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-o')
plt.plot((ours.loc[:,'backward-compute']+ours.loc[:,'forward-compute']).reset_index().iloc[:,-1], '-x')
plt.legend(['AMP '+'6-'*7+'6','FASOP 7'+('-6'*6)+'-5'])
plt.xlabel('GPU rank')
plt.ylabel('Per Iter. Time (ms)')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
for i in range(8):
    plt.axvline(i*8, color='k', linestyle='--')

plt.tight_layout()
plt.savefig('1248-computation.png')
# %%



f = open("11116-fasop",
"r")
doc = f.read()
data = yaml.load(doc, Loader=yaml.Loader)
fasop = pd.DataFrame(data)
# %%
fasop
# %%

for col in fasop.columns:
    print(col)
    plt.plot(fasop.loc[:,col].reset_index().iloc[:,-1], '-o')
    plt.legend(['fasop'])
    plt.title(col)
    plt.xlabel('[ gpu rank ]')
    plt.ylabel('[ elapsed time (ms) ]')
    plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
    # for i in range(16):
    #     plt.axvline(i*4, color='k', linestyle='--')
    plt.show()
# %%

plt.plot(fasop.loc[:,'post-process'].reset_index().iloc[:,-1], '-o')
plt.plot(fasop.loc[:,'forward-compute'].reset_index().iloc[:,-1], '-o')
plt.legend(['loss-compute','forward-compute'])
plt.title('loss-compute')
plt.xlabel('[ gpu rank ]')
plt.ylabel('[ elapsed time (ms) ]')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
# for i in range(16):
#     plt.axvline(i*4, color='k', linestyle='--')
plt.show()

# %%
plt.plot(fasop.loc[:,'loss-compute', ].reset_index().iloc[:,-1], '-o')
plt.plot(fasop.loc[:,'post-process', ].reset_index().iloc[:,-1], '-o')
plt.plot(fasop.loc[:,'forward-compute', ].reset_index().iloc[:,-1], '-o')
plt.legend(['loss-compute', 'post-process','forward-compute'])
plt.title('loss-compute')
plt.xlabel('[ gpu rank ]')
plt.ylabel('[ elapsed time (ms) ]')
plt.gca().yaxis.set_major_formatter(plt.FuncFormatter(lambda x, loc: "{:,}".format(int(x)) + ' ms'))
# %%
