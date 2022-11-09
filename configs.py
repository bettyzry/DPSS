
def getstep(dataname):
    if dataname == 'epilepsy':
        step = 206
    elif dataname == 'natops':
        step = 51
    elif dataname == 'racket_sports':
        step = 30
    else:
        print('ERROR')
    return step

intermediate_dir = './RS_z-intermediate_model_files/racket_sports/'