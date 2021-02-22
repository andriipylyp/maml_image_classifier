import os
if __name__ == "__main__":
    path_to_objectnet = 'D:'

    files = folders = 0
    if not os.path.isdir(path_to_objectnet+'/objectnet-map'):
        os.mkdir(path_to_objectnet+'/objectnet-map')
    for _, dirnames, _ in os.walk(path_to_objectnet+'\objectnet-1.0\objectnet-1.0\images'):
    # ^ this idiom means "we won't be using this value"
        for idx, name in enumerate(dirnames):
            with open('{0}/objectnet-map/map.csv'.format(path_to_objectnet), 'a') as file:
                if idx != len(dirnames)-1:
                    file.write(name+',')
                else:
                    file.write(name)
            
            for _, _, filenames in os.walk(path_to_objectnet+'\objectnet-1.0\objectnet-1.0\images\{0}'.format(name)):
                for idx2, filename in enumerate(filenames):
                    with open('{0}/objectnet-map/{1}.csv'.format(path_to_objectnet,idx), 'a') as file:
                        if idx2 != len(filenames)-1:
                            file.write('{0},'.format(filename))
                        else:
                            file.write(filename)