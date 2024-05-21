# Import critical packages
import matplotlib.pyplot as plt
from common_params import *
import pickle
import csv
from matplotlib.font_manager import findfont, FontProperties
import matplotlib as mpl


def fig_neuron_activation():

    # Note: to make this figure, you need to have run the forward model with these scenarios first
    # TODO need next line?
    # TODO scenarios = ['Uniform80R05', 'Uniform80R10', 'Uniform80R15']
    mpl.rcParams['font.family'] = 'Arial'

    espace = 0.85
    if espace == 0.85:
        e_txt = '085'
    elif espace == 1.1:
        e_txt = '110'
    else:
        e_txt = 'xxx'
    es_text = '_espace_' + e_txt

    font = findfont(FontProperties(family=['Arial']))
    activation_file = FWDOUTPUTDIR + 'neuronact_' + STD_TEXT + es_text + '.npz'
    print("Activation file: ", activation_file)
    data = np.load(activation_file, allow_pickle=True)
    surv_vals = data['arr_0']
    rpos_vals = data['arr_1']
    neuronact = data['arr_2']

    # can be useful for some debugging or draft figures
    hires = '_hi_res'
    descrip = "surv_" + str(np.min(surv_vals)) + "_" + str(np.max(surv_vals)) + "_rpos_" + \
              str(np.min(rpos_vals)) + "_" + str(np.max(rpos_vals)) + hires


    # params_file = FWDOUTPUTDIR + 'simParamssurv_0.04_0.9599999999999999_rpos_-0.95_0.9500000000000017_hi_res.npy'
    params_file = FWDOUTPUTDIR + 'simParams' + descrip
    sp = np.load(params_file + '.npy', allow_pickle=True)
    # with open(params_file, 'rb') as f:
    #     sp = pickle.load(f)

    elec_2d = 7  # which electrode was used in the 2D forward model
    posvals = np.arange(0, 33, 0.01) - 14.6 - (elec_2d * espace)

    # Load monopolar data
    datafile = FWDOUTPUTDIR + "Monopolar_2D_" + STD_TEXT + es_text + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        mono_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            mono_thr[i, :] = row

    # Load tripolar data
    datafile = FWDOUTPUTDIR + "Tripolar_09_2D_" + STD_TEXT + es_text + ".csv"
    file = open(datafile)
    numlines = len(file.readlines())
    file.close()

    with open(datafile, newline='') as csvfile:
        datareader = csv.reader(csvfile, delimiter=',')
        ncol = len(next(datareader))
        csvfile.seek(0)
        tripol_thr = np.empty([numlines, ncol])
        for i, row in enumerate(datareader):
            # Do the parsing
            tripol_thr[i, :] = row

    # Make plots
    survidxs = []
    rposidxs = []
    plt_surv_vals = [0.8]  # desired survival values to be plotted
    nsurv = len(plt_surv_vals)
    plt_rpos_vals = [-0.5, 0.0, 0.5]  # desired rpos values to be plotted
    nrpos = len(plt_rpos_vals)
    fig, axs = plt.subplots(nrpos, nsurv, sharex=True, sharey=True, figsize=(5, 8))
    plt.figtext(0.03, 0.88, 'A', fontsize=20, weight='bold')
    plt.figtext(0.03, 0.62, 'B', fontsize=20, weight='bold')
    plt.figtext(0.03, 0.33, 'C', fontsize=20, weight='bold')

    for i, val in enumerate(plt_surv_vals):
        theidx = np.argmin(np.abs(surv_vals - val))
        survidxs.append(theidx)

    for i, val in enumerate(plt_rpos_vals):
        theidx = np.argmin(np.abs(rpos_vals - val))
        rposidxs.append(theidx)

    # Set labels and plot
    titletext = []
    for i, ax in enumerate(axs.flat):
        row = int(i / nsurv)
        col = int(np.mod(i, nsurv))
        # if row == 0:
        #     titletext = 'surv = %.2f' % surv_vals[survidxs[col]]
        #     ax.set_title(titletext)

        n_p_c = sp['neurons']['neur_per_clust']
        ax.plot(posvals + espace, neuronact[survidxs[col], rposidxs[row], 0, 0, :]/n_p_c, '.',
                color='blue', linewidth=0.5)
        ax.plot(posvals + espace, neuronact[survidxs[col], rposidxs[row], 1, 0, :]/n_p_c, '.',
                color='red', linewidth=0.5)
        ax.set(xlabel='Longitudinal distance (mm)')
        if row == 1 and col == 0:
            ax.set(ylabel='Fractional neuronal activation')
        # place threshold values here
        xlimit = [-4, 4]
        ax.set_xlim((xlimit[0], xlimit[1]))
        ax.set_yticks([0, 0.1, 0.2, 0.3])
        ax.spines.right.set_visible(False)
        ax.spines.top.set_visible(False)
        ax.label_outer()
        if col == (nsurv - 1):
            mytext = 'Distance = ' + str(1 - plt_rpos_vals[row]) + ' mm'
            ax.text(4.0, 0.3, mytext, horizontalalignment='right')
            m_thr_text = "Monopolar thr.: %.2f dB" % mono_thr[survidxs[col], rposidxs[row]]
            t_thr_text = "Tripoloar thr.: %.2f dB" % tripol_thr[survidxs[col], rposidxs[row]]
            ax.text(4.0, 0.25, m_thr_text, horizontalalignment='right')
            ax.text(4.0, 0.2, t_thr_text, horizontalalignment = 'right')

            print('surv ', titletext, ' dist ', str(1 - plt_rpos_vals[row]), ' Thr: ', mono_thr[survidxs[col],
                        rposidxs[row]], ' and ', tripol_thr[survidxs[col], rposidxs[row]])
    # plt.show()

    # Save figure
    fig_filename = 'fig3_neuronact.pdf'
    plt.savefig(fig_filename, format='pdf')
    fig_filename = 'fig3_neuronact.png'
    plt.savefig(fig_filename, format='png')

    plt.show()


if __name__ == '__main__':

    fig_neuron_activation()
