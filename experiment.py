import pandas as pd

import StreamRunner


def main():
    funclist = ['gdn']
    # run_mod = ['F', 'skopt', 'skopt2', 'DPSS']
    run_mod = ['skopt', 'skopt2', 'DPSS']
    # dataname = 'epilepsy'
    dataname = 'racket_sports'
    output = '%s_gdn_add2.csv' % dataname
    runs = 1
    # f = open(output, 'w')
    # f.write('')
    # f.close()
    for datanum in range(4, 5):
        for func in funclist:
            # func = 'sesd'
            for mod in run_mod:
                if mod == 'skopt':
                    do_init = True
                    do_refit = False
                    refit = 'skopt'
                elif mod == 'F':
                    do_init = False
                    do_refit = False
                    refit = 'F'
                elif mod == 'skopt2':
                    do_init = True
                    do_refit = True
                    refit = 'skopt'
                elif mod == 'DPSS':
                    do_init = True
                    do_refit = True
                    refit = 'DPSS'
                else:
                    print('ERROR')
                    return
                for run in range(runs):
                    print('%s_%d_%s_%s_%d' % (dataname, datanum, func, mod, run))
                    model = StreamRunner.main(func=func, dataname=dataname, datanum=datanum, do_refit=do_refit, refit_func=refit, fit_size=20000, do_init=do_init)
                    result = model.eval_result['f1']
                    f = open(output, 'a')
                    f.write('%d_%s_%s,' % (datanum, func, refit))
                    for r in result:
                        f.write('%f,' % r)
                    f.write('\n')
                    f.close()
    return


if __name__ == '__main__':
    print('lalala')
    main()