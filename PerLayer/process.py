import re
import argparse
import sys

parser = argparse.ArgumentParser()
parser.add_argument('--output', type=str, default='./output', help='The output of DebugDumpGroup', required=True)
args = parser.parse_args()

def getRawData(path):
    with open(path,'r') as f:
        RawData = []
        found=False
        lines = f.readlines()
        for line in lines:
            if(re.search('-->Begin to DebugDumpGroup<--', line)):
                found=True
            if(found):
                RawData.append(line)
            if(re.search('-->End of DebugDumpGroup<--', line)):
                found=False
        return RawData

def getData(RawData):
    data=[]
    for line in RawData:
        if(re.search('%[0-9]* = ', line)):
            line=line.strip('\n')
            num_beg=line.find('%')
            num_end=line.find('=')
            num=line[num_beg:num_end]
            line = line[num_end+2:]
            layer=[]
            tmp=[]
            tmp.append(num)
            layer.append(tmp)
            tmp=[]
            num_name_end=line.find('(')
            tmp.append(line[:num_name_end])
            layer.append(tmp)
            tmp=[]
            line = line[num_name_end:]
            num_input_end=line.find(')')
            tmp.append(line[:num_input_end+1])
            layer.append(tmp)
            #print(line)
            #cnt=0
            tmp=[]
            while line.find('group=')>=0:
                #print(line)
                beg=line.find('group=')
                end=line.find(' */')
                group=line[beg:end]
                tmp.append(group)
                line=line[end+3:]
                #print(line)
                #if cnt>2:
                #    break
                #cnt+=1
            layer.append(tmp)
            data.append(layer)

    return data

def process(data):
    group_id={'example':0}
    ids=[]
    for lines in data:
        #print(line)
        #print()
        print('layer id    : '+str(lines[0]))
        print('layer op    : '+str(lines[1]))
        print('layer input : '+str(lines[2]))
        #print('layer group_id : '+str(lines[3]))
        # id op input group
        l0 = lines[0]
        l1 = lines[1]
        l2 = lines[2]
        l3 = lines[3]
        groups=l3
        if(len(groups)==3):
            assert(groups[0]==groups[1] and groups[1]!=groups[2])
        if group_id.__contains__(groups[-1]):
            ids.append(group_id[groups[-1]])
        else:
            ids.append(l0)
            group_id[groups[-1]]=l0
        print('layer group : '+str(group_id[groups[-1]]))
        print()
        
    for i in range(len(ids)):
        data[i].append(ids[i])
    
    dominator=[]
    for line in data:
        #print(line[0][0])
        #print(line[4][0])
        if(line[0][0]==line[4][0]):
            dominator.append(line[0][0])
    
    print('Dominator node : ' + str(dominator) +'\n' )
    layer_fuse_name=[]
    for dom in dominator:
        name=''
        for line in data:
            if dom == line[4][0]:
                name=name+'_'+line[1][0]
        layer_fuse_name.append(name)
        
    for line in (layer_fuse_name):
        print(line)
    print('len of data is : '+str(len(data)))


def main():
    path_output = args.output
    RawData = getRawData(path_output)
    #print(RawData)
    data = getData(RawData)
    #print(data)
    process(data)

if __name__ == '__main__':
    main()



