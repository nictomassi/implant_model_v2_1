# Forward Model version 4. 25 October 2023

import pickle
import datetime
import os
import csv
import matplotlib.pyplot as plt
#  These are local to this project
import set_scenario as s_scen
import surv_full
import get_thresholds as gt
from common_params import *


def fwd_model_4(mode):

    if mode == 'main':
        pass  # proceed as normal
    elif mode == 'survey':
        # load survey params
        param_file = 'surv_params.txt'
        tempdata = np.zeros(4)  # 4 values
        if os.path.exists(param_file):
            with open(param_file, newline='') as csvfile:
                datareader = csv.reader(csvfile, delimiter=',')
                csvfile.seek(0)
                for i, row in enumerate(datareader):
                    # Do the parsing
                    tempdata[i] = row[0]
        NEURONS['act_stdrel'] = tempdata[1]
        NEURONS['act_ctr'] = tempdata[2]
        NEURONS['thrtarg']=100
        espace = tempdata[3]  # should be overridden by scenario/subject
    elif mode == 'gui':
        # read CSV file with position, survival espace
        print('foo')
        # scenarios = ['gui_scenario']
    else:  # should not happen
        print('fwd_model called with unrecognized mode: ', mode)
        exit()

    # We depend on voltage and activation tables calculated using
    # voltage_calc.py and saved as a .dat file. the file is specified in common_params.py
    with open(FIELDTABLE, 'rb') as combined_data:
        data = pickle.load(combined_data)
        combined_data.close()

    fp = data[0]
    fp['zEval'] = np.array(fp['zEval'])
    #  v_vals = data[1] #  voltage values not used
    act_vals = data[2]
    GRID['table'] = act_vals

    COCHLEA['res1'] = fp['resInt'] * np.ones(NELEC)  # Note these values do not match those of Goldwyn et al., 2010
    COCHLEA['res2'] = fp['resExt'] * np.ones(NELEC)  # resistivities are in Ohm*cm (conversion to Ohm*mm occurs later)
    GRID['r'] = fp['rspace']  # only 1 of the 3 cylindrical dimensions can be a vector (for CYLINDER3D_MAKEPROFILE)

    if_plot = True  # Whether to plot the results
    n_sig = len(sigmaVals)

    # Automatically create scenarios with uniform conditions across electrode positions
    # We've mostly abandoned this for customized scenarios from set_scenario()
    # survScenarios = [ 75]
    # rposScenarios = [-0.5, 0.0, 0.5]

    # nSurvS = len(survScenarios)
    # nRposS = len(rposScenarios)

    # nScenarios = nSurvS*nRposS
    # scenarios = []

    # for i in range(0, nSurvS):
    #     for j in range(0, nRposS):
    #         tempval = str(rposScenarios[j])
    #         print('tempval = ', tempval)
    #         scenarios.append('Uniform'+str(survScenarios[i])+'R'+tempval.replace('.', ''))

    for scenario in scenarios:
        # if this scenario is a subject, set use_forward_model to be false
        first_let = scenario[0]
        if (first_let == 'A' or first_let == 'S') and scenario[1:3].isnumeric():
            print('This scenario, ', scenario, ' appears to be for a subject, not a forward model scenario. Skipping.')
            continue

        if mode == 'gui':
            # read CSV file
            param_file = 'guifile.csv'
            if os.path.exists(param_file):
                with open(param_file, newline='') as csvfile:
                    datareader = csv.reader(csvfile, delimiter=',')
                    surv_vals = np.array(next(datareader), dtype='float')
                    electrodes['rpos'] = np.array(next(datareader), dtype=float)  # Do the parsing
                    espace = float(next(datareader)[0])

        else:
            [surv_vals, electrodes['rpos'], espace] = s_scen.set_scenario(scenario, NELEC)


        elec_midpoint = GRID['z'][-1]/2.0  # electrode array midpoint
        array_base = -(np.arange(NELEC - 1, -1, -1) * espace)
        array_mid = (array_base[0] + array_base[-1])/2.0
        # electrodes['zpos'] = ELEC_BASALPOS - (np.arange(NELEC - 1, -1, -1) * espace)

        electrodes['zpos'] = (elec_midpoint - array_mid) + array_base

        if not os.path.isdir(FWDOUTPUTDIR):
            os.makedirs(FWDOUTPUTDIR)

        outfile = FWDOUTPUTDIR + 'FwdModelOutput_' + scenario + '.csv'

        # Additional setup
        RUN_INFO['scenario'] = scenario
        RUN_INFO['run_time'] = datetime.datetime.now()
        COCHLEA['radius'] = np.ones(NELEC) * fp['cylRadius']  # note that '.rpos' is in mm, must fit inside the radius

        # Construct the simParams structure
        simParams['cochlea'] = COCHLEA
        simParams['electrodes'] = electrodes
        simParams['channel'] = CHANNEL
        simParams['grid'] = GRID
        simParams['run_info'] = RUN_INFO

        # Example of neural activation at threshold (variants of Goldwyn paper) ; using choices for channel, etc.
        # made above.
        # Keep in mind that the activation sensitivity will be FIXED, as will the number of neurons required
        # for threshold. Therefore, the final current to achieve threshold will vary according to the simple
        # minimization routine.
        avec = np.arange(0, 1.01, .01)  # create the neuron count to neuron spikes transformation
        rlvec = NEURONS['coef'] * (avec ** 2) + (1 - NEURONS['coef']) * avec
        rlvec = NEURONS['neur_per_clust'] * (rlvec ** NEURONS['power'])
        rlvltable = np.stack((avec, rlvec))  # start with the e-field(s) created above, but remove the current scaling

        # Specify which variables to vary and set up those arrays
        thr_sim_db = np.empty((NELEC, n_sig))  # Array for threshold data for different stim elecs and diff sigma values
        thr_sim_db[:] = np.nan

        # Get survival values for all 330 clusters from the 16 values at electrode positions.
        NEURONS['nsurvival'] = surv_full.surv_full(simParams['electrodes']['zpos'], surv_vals, simParams['grid']['z'])
        NEURONS['rlvl'] = rlvltable
        simParams['neurons'] = NEURONS
        #  Sanity check. Could add other sanity checks here
        if simParams['grid']['r'] < 1:
            raise ('Ending script. One or more evaluation points are inside cylinder;\
             not appropriate for neural activation.')

        # Determine threshold for each value of sigma
        for i in range(0, n_sig):  # number of sigma values to test
            simParams['channel']['sigma'] = sigmaVals[i]
            [thr_sim_db[:, i], neuron_vals] = gt.get_thresholds(act_vals, fp, simParams)

        # Write a csv file
        with open(outfile, mode='w') as data_file:
            data_writer = csv.writer(data_file, delimiter=',', quotechar='"')
            for row in range(0, NELEC):
                data_writer.writerow([row, surv_vals[row], electrodes['rpos'][row], thr_sim_db[row, 0],
                                      thr_sim_db[row, 1]])
        data_file.close()

        # Save simParams
        spname = FWDOUTPUTDIR + 'simParams' + scenario
        with open(spname + '.pickle', 'wb') as f:
            pickle.dump(simParams, f, pickle.HIGHEST_PROTOCOL)
        print('saved: ', spname + '.pickle')
        # Note that this is saving only the last simParams structure from the loops on sigma and in get_thresholds.

        # Plot the results, if desired
        if if_plot:
            fig1, ax1 = plt.subplots()
            ax1.plot(np.arange(0, NELEC) + 1, thr_sim_db, marker='o')
            title_text = 'Threshold ' + scenario
            ax1.set(xlabel='Electrode number', ylabel='Threshold (dB)', title=title_text)

            plt.show()

    # Save PDF, if desired
    #        legend([simSurv, targetSurv], 'sim', 'target', 'Location', 'north');
    #        print('-dpdf', '-painters', '-bestfit', 'epsFig.pdf');
    #        movefile('epsFig.pdf', [FWDOUTPUTDIR scenario '_thresh.pdf']);


if __name__ == '__main__':
    fwd_model_4('main')  # alternatives are 'main', 'gui' and "survey'
