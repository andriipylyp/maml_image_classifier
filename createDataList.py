import os
import pandas as pd
if __name__ == "__main__":
    path_to_objectnet = 'D:'

    files = folders = 0
    if not os.path.isdir(path_to_objectnet+'/objectnet-map'):
        os.mkdir(path_to_objectnet+'/objectnet-map')
    class_list = []
    images = []
    for _, dirnames, _ in os.walk(path_to_objectnet+'\objectnet-1.0\objectnet-1.0\images'):
    # ^ this idiom means "we won't be using this value"
        for name in dirnames:
            class_list.append(name)
            image_buf = []
            for _, _, filenames in os.walk(path_to_objectnet+'\objectnet-1.0\objectnet-1.0\images\{0}'.format(name)):
                for filename in filenames:
                    image_buf.append(filename)
            images.append(image_buf)
    buf_dict = {}
    for idx,c in enumerate(class_list):
        buf_dict[c] = images[idx]
    
    df = pd.DataFrame(dict([ (k, pd.Series(v)) for k,v in buf_dict.items()]))
    df.to_csv(path_to_objectnet+'/objectnet-map/map.csv',index=False)