import pandas as pd
import numpy as np
import random
import matplotlib.pyplot as plt
import os


def split(train, test, ytrain, ytest):
    walking = []
    running = []
    sawing = []
    epilepsy = []

    for ii, act in enumerate(ytrain):
        if act == 'WALKING':
            walking.append(train[ii])
        elif act == 'RUNNING':
            running.append(train[ii])
        elif act == 'SAWING':
            sawing.append(train[ii])
        elif act == 'EPILEPSY':
            epilepsy.append(train[ii])
        else:
            print("ERROR!")

    for ii, act in enumerate(ytest):
        if act == 'WALKING':
            walking.append(test[ii])
        elif act == 'RUNNING':
            running.append(test[ii])
        elif act == 'SAWING':
            sawing.append(test[ii])
        elif act == 'EPILEPSY':
            epilepsy.append(test[ii])
        else:
            print("ERROR!")

    d = {
        '0': walking,
        '1': running,
        '2': sawing,
        'epilepsy': epilepsy
    }

    return d


def main():
    data_path = './data/epilepsy/'
    train = np.load(data_path + 'train_array.npy')
    ytrain = np.load(data_path + 'train_label.npy')
    test = np.load(data_path + 'test_array.npy')
    ytest = np.load(data_path + 'test_label.npy')

    dic = split(train, test, ytrain, ytest)
    outlier_data = dic['epilepsy']
    for kk in range(10):
        output = np.array([[0,0,0]])
        pattern = np.array([])

        for ii in range(10):
            norm_pattern = random.randint(0, 2)
            norm_data = dic[str(norm_pattern)]
            norm_num = random.randint(50, 100)
            outlier_num = random.randint(5, 10)
            outlier_loc = np.random.choice(norm_num, outlier_num, replace=False)
            for jj in range(norm_num):
                if jj in outlier_loc:
                    num = random.randint(0, len(outlier_data) - 1)
                    data = outlier_data[num]
                    p = 3
                else:
                    num = random.randint(0, len(norm_data)-1)
                    data = norm_data[num]
                    p = norm_pattern
                output = np.vstack((output, data))
                pattern = np.concatenate((pattern, np.array([p for i in range(206)])))

        print(output)
        df = pd.DataFrame(output, columns=['v1', 'v2', 'v3'])
        df = df.drop([0], axis=0)
        df['pattern'] = pattern
        df['label'] = 0
        df['label'].loc[df['pattern'] == 3] = 1
        df['index'] = [i for i in range(len(df))]
        df.to_csv('./data/epilepsy_c/epilepsy_%d.csv' % kk, index=False)


def split_plot():
    for kk in range(10):
        outpath = './figs/epilepsy'
        path = './data/epilepsy_c/epilepsy_%d.csv' % kk
        df = pd.read_csv(path)[:10000]

        ax1 = plt.subplot(3, 1, 1)
        plt.plot(df['v1'].values, 'b--')
        plt.ylabel('y1')
        ax = ax1.twinx()
        color = 'tab:blue'
        ax.set_ylabel('label', color=color, size=14)
        ax.plot(df['label'].values, color=color)

        ax2 = plt.subplot(3, 1, 2)
        plt.plot(df['v2'].values, 'g--')
        plt.ylabel('y2')
        ax = ax2.twinx()
        color = 'tab:blue'
        ax.set_ylabel('label', color=color, size=14)
        ax.plot(df['label'].values, color=color)

        ax3 = plt.subplot(3, 1, 3)
        plt.plot(df['v3'].values, 'r--')
        plt.ylabel('y2')
        ax = ax3.twinx()
        color = 'tab:blue'
        ax.set_ylabel('label', color=color, size=14)
        ax.plot(df['label'].values, color=color)
        plt.show()
        return
        # if not os.path.exists(outpath):
        #     os.makedirs(outpath)
        # plt.savefig(os.path.join(outpath, "epilepsy_%d.png" % kk))
        # plt.close()


def epilepsy_plot_fig():
    for kk in range(10):
        datapath = './data/epilepsy_c/epilepsy_%d.csv' % kk
        outpath = './figs/epilepsy'
        df = pd.read_csv(datapath)

        fig, ax1 = plt.subplots(figsize=(20, 5))
        ax1.set_ylabel('v1', size=14)
        ax1.plot(df['v1'].values)
        ax1.set_ylabel('v2', size=14)
        ax1.plot(df['v2'].values)
        ax1.set_ylabel('v3', size=14)
        ax1.plot(df['v3'].values)
        ax1.tick_params(axis='y', labelsize=12)
        ax2 = ax1.twinx()
        color = 'tab:blue'
        ax2.set_ylabel('label', color=color, size=14)
        ax2.plot(df['label'].values, color=color)
        ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
        ax2.grid('off')
        ax2.set_ylim(0, 2)
        plt.title("epilepsy_%d.csv" % kk)
        plt.show()
        if not os.path.exists(outpath):
            os.makedirs(outpath)
        plt.savefig(os.path.join(outpath, "epilepsy_%d.png" % kk))
        plt.close()


def plot():
    datapath = './data/test.csv'
    outpath = './figs'
    df = pd.read_csv(datapath)
    df = df[:500]
    fig, ax1 = plt.subplots(figsize=(20, 5))
    ax1.plot(df.values)
    ax1.tick_params(axis='y', labelsize=12)
    ax2 = ax1.twinx()
    color = 'tab:blue'
    ax2.set_ylabel('class', color=color, size=14)
    ax2.plot(df['class'].values, color=color)
    ax2.tick_params(axis='y', labelcolor=color, labelsize=12)
    ax2.grid('off')
    ax2.set_ylim(0, 2)
    plt.title("test.csv")
    plt.show()
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    plt.savefig(os.path.join(outpath, "test.png"))
    plt.close()


if __name__ == '__main__':
    # main()
    # epilepsy_plot_fig()
    # plot()
    split_plot()