import os

data_path = os.path.expanduser('~/project-data')
if __name__=='__name__':
    if not os.path.exists(data_path):
        print('making data directory...')
        os.mkdir(data_path)

    download_path = os.path.join(data_path, 'graph-dataset.zip')
    if not os.path.exists(download_path):
        print('downloading dataset...')
        source = 'https://download.microsoft.com/download/7/8/4/784863FE-8883-4840-8228-211ABE680364/graph-dataset.zip'
        os.system('wget -O %s %s' % (download_path, source))
        print('finish downloading')


