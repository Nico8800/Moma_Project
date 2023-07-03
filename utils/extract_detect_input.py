import numpy as np
import pandas as pd
import json
import os
#we re going to build a json file to use during the data loading of the detection module

train_dic = {'index':[],
    'raw_video_id':[],
    'trim_video_id':[],
    'frame_id':[],
    'path':[],
    'bboxes':[],
    'classes':[],
    'id_in_video':[],
    'class_label':[]
    }

val_dic = {'index':[],
    'raw_video_id':[],
    'trim_video_id':[],
    'frame_id':[],
    'path':[],
    'bboxes':[],
    'classes':[],
    'id_in_video':[],
    'class_label':[]
    }

frame_root='/home/nicolasg/moma_data/frame_data'

# we're gonna access the annotations to build the csv files
graph_ann_file='/home/nicolasg/code/annotations/anns/graph_anns.json'

# in order to have the correct split we have to acces 

train_path='/home/nicolasg/code/annotations/anns/split_by_trim/train.txt'
val_path='/home/nicolasg/code/annotations/anns/split_by_trim/val.txt'

with open(train_path) as f:
    lines = f.readlines()
train_text=[i[:-1] for i in lines]

with open(val_path) as f:
    lines = f.readlines()
val_text=[i[:-1] for i in lines]

# we must also attribute a label to each and every object/actor. We will consider the object than the
# actors in the enumeration order. 
actor_file='/home/nicolasg/code/annotations/anns/actor_cnames.txt'
object_file='/home/nicolasg/code/annotations/anns/object_cnames.txt'

with open(object_file) as f:
    object_lines=f.readlines() 
with open(actor_file) as f:
    actor_lines=f.readlines()

actors_list=[i[:-1] for i in actor_lines]
actors_list[-1]=actors_list[-1]+'r'
objects_list=[i[:-1] for i in object_lines]
objects_list[-1]=objects_list[-1]+'s'
entity_list=objects_list+actors_list

with open(graph_ann_file) as f:
    graph_anns=json.load(f)

# we now are going to iterate through every frame and build consequently the dictionnaries
train_index=0
val_index=0

for i, dico in enumerate (graph_anns): # we check out each frame
    print(i,'/',len(graph_anns))
    trim_video_id = dico['trim_video_id'] # we get where each frame is from (trim_clip))

    if str(trim_video_id) in train_text:
        train_dic['index'].append(train_index)
        train_index+=1
        train_dic['trim_video_id'].append(trim_video_id)
        raw_video_id=dico['raw_video_id']
        train_dic['raw_video_id'].append(raw_video_id)
        train_dic['frame_id'].append(dico['graph_id'])
        train_dic['path'].append(os.path.join(frame_root,trim_video_id,str(dico['graph_id']-1)+'.jpg'))
        train_dic['bboxes'].append([])
        train_dic['classes'].append([])
        train_dic['id_in_video'].append([])
        train_dic['class_label'].append([])
        blackscreen=True # black screen corresponds to no object
        # we re now going to add the actors and objects'labels (bbox and class).
        for object in dico['annotation']['objects']:
            # the format of the bbox is the following: x_min, y_min, x_max, y_max
            train_dic['bboxes'][-1].append([object['bbox']['bottomLeft']['x']-1,
                                            object['bbox']['topLeft']['y']-1,
                                            object['bbox']['topRight']['x']-1,
                                            object['bbox']['bottomRight']['y']-1])
            train_dic['classes'][-1].append(entity_list.index(object['class']))
            train_dic['id_in_video'][-1].append(object['id_in_video'])
            train_dic['class_label'][-1].append('object')
            blackscreen=False
        for actor in dico['annotation']['actors']:
            # the format of the bbox is the following: x_min, y_min, x_max, y_max
            train_dic['bboxes'][-1].append([actor['bbox']['bottomLeft']['x']-1,
                                            actor['bbox']['topLeft']['y']-1,
                                            actor['bbox']['topRight']['x']-1,
                                            actor['bbox']['bottomRight']['y']-1])
            train_dic['classes'][-1].append(entity_list.index(actor['class']))
            train_dic['id_in_video'][-1].append(actor['id_in_video'])
            train_dic['class_label'][-1].append('actor')
            blackscreen=False
        if blackscreen:
            train_dic['bboxes'][-1].append([0,
                                            0,
                                            312,
                                            312])#crop size
            train_dic['classes'][-1].append(140)
            train_dic['class_label'][-1].append('background')
    else:
        blackscreen=True
        val_dic['index'].append(val_index)
        val_index+=1
        val_dic['trim_video_id'].append(trim_video_id)
        raw_video_id=dico['raw_video_id']
        val_dic['raw_video_id'].append(raw_video_id)
        val_dic['frame_id'].append(dico['graph_id']-1)
        val_dic['path'].append(os.path.join(frame_root,trim_video_id,str(dico['graph_id']-1)+'.jpg'))
        val_dic['bboxes'].append([])
        val_dic['classes'].append([])
        val_dic['id_in_video'].append([])
        val_dic['class_label'].append([])
        val_dic['area'].append([])
        # we re now going to add the actors and objects'labels (bbox and class).
        for object in dico['annotation']['objects']:
            # the format of the bbox is the following: x_min, y_min, x_max, y_max
            val_dic['bboxes'][-1].append([object['bbox']['bottomLeft']['x']-1,
                                            object['bbox']['topLeft']['y']-1,
                                            object['bbox']['topRight']['x']-1,
                                            object['bbox']['bottomRight']['y']-1])
            val_dic['classes'][-1].append(entity_list.index(object['class']))
            val_dic['id_in_video'][-1].append(object['id_in_video'])
            val_dic['class_label'][-1].append('object')
            blackscreen=False
        for actor in dico['annotation']['actors']:
            # the format of the bbox is the following: x_min, y_min, x_max, y_max
            val_dic['bboxes'][-1].append([actor['bbox']['bottomLeft']['x']-1,
                                            actor['bbox']['topLeft']['y']-1,
                                            actor['bbox']['topRight']['x']-1,
                                            actor['bbox']['bottomRight']['y']-1])
            val_dic['classes'][-1].append(entity_list.index(actor['class']))
            val_dic['id_in_video'][-1].append(actor['id_in_video'])
            val_dic['class_label'][-1].append('actor')
            blackscreen=False
        if blackscreen:
            val_dic['bboxes'][-1].append([0,
                                            0,
                                            312,
                                            312])#crop size
            val_dic['classes'][-1].append(140)
            val_dic['class_label'][-1].append('background')


train_json=json.dumps(train_dic)
val_json=json.dumps(val_dic)

f = open('/home/nicolasg/code/annotations/train_detect_mod.json','w')
f.write(train_json)
f.close()
f = open('/home/nicolasg/code/annotations/val_detect_mod.json','w')
f.write(val_json)
f.close()