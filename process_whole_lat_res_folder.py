#
#
###################################################################
#####                                                        ######
#####       N42.59 Lateral Resolution Python Package         ######
#####       written by Jack Glover                           ######
#####       Date: 2024-03-28                                 ######
#####       Version: 0.7                                     ######
#####                                                        ######
###################################################################
#
#
# DISCLAIMER: This code is still under development so may have bugs
# and may need to be improved in some ways
#
# An example call of this code is given at the bottom of this file.
# You must specify a folder that contains images of the test objects.
# The images of the test object must:
#     - be cropped to the edges of the test object
#     - have filename(s) ending with "_A" or "_large" for test obj A
#     - have filename(s) ending with "_B" or "_small" for test obj B
# The code processes those images and produces various informative
# plots, images, and output files.
#
# The code should work on Python 3.7 to 3.10 and likely other versions
# too, once the proper libraries are installed.
#
# Things about the code that need to be documented or corrected:
# - It would be nice to provide a good way to override the fiducial
#   determinations in cases where they have gone wrong. 
# - The R&S data can be really tricky to process for the prototype 
#   test object because the fiducials aren't always very dark
#
#
#
#
#
#
#
#


import os
import numpy as np
import pandas as pd
import glob
from scipy import interpolate
import matplotlib.pyplot as plt
from process_lat_res_test_image_N4259 import process_lat_res_test_image

from matplotlib.backends.backend_pdf import PdfPages


def process_whole_lat_res_folder(folder,img_str,f_diff_cutoff=None, aureole=False, PDF_obj=None, MTF20_all=False):
    print('Processing folder: '+folder)
    img_files = glob.glob(folder+img_str)



    if PDF_obj is not None:
        pdf_filename_summary = folder+'N4259_lateral_resolution_results_summary.pdf'
        PDF_obj = PdfPages(pdf_filename_summary)

    MTFs     = []
    MTF_errs = []
    f_arrs   = []

    plt.rcParams.update({'errorbar.capsize': 2})

    # Loop over all images
    for img_filename in img_files:
        full_basename = img_filename.split('.')[-2]
        basename      = full_basename.split('\\')[-1]
        test_obj_size = basename.split('_')[-1]

        if test_obj_size == "large": test_obj_size = 'A'
        elif test_obj_size == "small": test_obj_size = 'B'

        print('    file: '+img_filename+',  (test object '+test_obj_size+')')

        # This info csv file has information about the layout of the test object (A or B or some alternate version)
        info_filename = 'test_object_line_frac_' + test_obj_size + '.csv'
        if os.path.isfile(info_filename): info_csv = pd.read_csv(info_filename)
        else:
            print('The filename should end with "_A" or "_B".')
            print('In this case, the filename ended in "'+test_obj_size+'"')
            print("and the corresponding info file wasn't found "+info_filename)
            raise NameError(info_filename)

        plot_folder = folder + basename + '/'

        f_arr, MTF_arr, MTF_arr_err = \
            process_lat_res_test_image(img_filename,info_csv, plot_folder=plot_folder,
                                        f_diff_cutoff=f_diff_cutoff,aureole=aureole,PDF_obj=PDF_obj)
        MTFs.append(MTF_arr)
        MTF_errs.append(MTF_arr_err)
        f_arrs.append(f_arr)



    #   Main MTF plot
    fig = plt.figure(figsize=(8, 5))
    MTFs_all = np.concatenate(MTFs)
    MTF_errs_all = np.concatenate(MTF_errs)
    f_arrs_all = np.concatenate(f_arrs)
    f_unique = np.unique(f_arrs_all)
    f_unique.sort()
    MTF_ave_arr = f_unique*0.0
    MTF_stdev_arr = f_unique * 0.0
    for i,f in enumerate(f_unique):
        ind = (f_arrs_all==f)
        MTF_f = MTFs_all[ind]
        MTF_ave_arr[i] = np.mean(MTF_f)
        MTF_stdev_arr[i] = np.std(MTF_f)

    plt.errorbar(f_unique, MTF_ave_arr, yerr=MTF_stdev_arr,fmt='.',label=full_basename,color='black')

    f_max = np.min(f_unique[MTF_ave_arr < 0.05])
    plt.xlim([0,f_max])

    MTF_model = interpolate.interp1d(MTF_ave_arr,f_unique)
    MTF20 = MTF_model([0.2])

    plt.plot([MTF20],[0.2],'*',color='gray')
    #plt.plot([MTF20,MTF20],[0,1],'--',color='gray')
    plt.title(f'Folder {folder} MTF20 = {MTF20[0]:.3f} cy/mm')

    plt.annotate(f'MTF20', xy=(MTF20, 0.2), xytext=(30, 30),
                textcoords='offset points', ha='center', va='bottom',
                arrowprops=dict(arrowstyle='->', connectionstyle='arc3,rad=0',
                                color='gray'))
    plt.ylim([0,1])
    plt.xlabel('Frequency  (cy/mm)')
    plt.ylabel('Modulation Transfer Function (MTF)')
    if PDF_obj is None:
        plt.savefig(folder + 'MTF_vs_f_all_test_patterns.png')
    else:
        PDF_obj.savefig()
    plt.close()

    if False:
        #   plotting each point with crappy error bars MTF plot
        fig = plt.figure(figsize=(10, 6))
        #plt.xlim([0,0.15])
        for i,f_arr in enumerate(f_arrs):
            MTF_arr = MTFs[i]
            MTF_arr_err = MTF_errs[i]
            plot_ind = MTF_arr > -1
            f_arr = f_arrs[i]
            full_basename = img_files[i].split('.')[-2].split('\\')[-1]
            plt.errorbar(f_arr[plot_ind], MTF_arr[plot_ind], yerr=MTF_arr_err[plot_ind],
                         fmt='x',label=full_basename,color='black')
        if MTF20_all:
            f_all = np.concatenate(f_arrs)
            MTF_all = np.concatenate(MTFs)
            f_max = np.min(f_all[MTF_all < 0.000001])
            MTF_model = interpolate.interp1d(MTF_all[f_all<f_max*1.01],f_all[f_all<f_max*1.01])
            MTF20 = MTF_model([0.2])

            plt.plot([MTF20],[0.2],'o',color='red')
            plt.plot([MTF20,MTF20],[0,1],'--',color='gray')
            plt.title(f'Folder {folder} MTF20 = {MTF20[0]:.2f} cy/mm')
        else: plt.title(folder)

        if f_diff_cutoff != None:
            if False:
                f_model,MTF_model = create_MTF_model(0.25, aureole=aureole, f_diff_cutoff=f_diff_cutoff)
                plt.plot(f_model,MTF_model,'-',label='model')
                plt.ylim([0, 1])

        plt.ylim([0,1])
        plt.xlim(xmin=0)
        plt.legend(prop={'size': 6})
        plt.xlabel('Frequency  (cy/mm)')
        plt.ylabel('MTF')
        if PDF_obj is None:
            plt.savefig(folder + 'MTF_vs_f_all_test_patterns2.png')
        else:
            PDF_obj.savefig()
        plt.close()


    if False:
        fig = plt.figure(figsize=(10, 6))
        plt.title(folder)
        for i,f_arr in enumerate(f_arrs):
            MTF_arr = MTFs[i]
            MTF_arr_err = MTF_errs[i]
            plot_ind = MTF_arr > -0.01
            w_arr = 1.0 / (2 * f_arrs[i])
            full_basename = img_files[i].split('.')[-2].split('\\')[-1]
            plt.errorbar(w_arr[plot_ind], MTF_arr[plot_ind], yerr=MTF_arr_err[plot_ind], fmt='x',label=full_basename)

        plt.ylim([0,1])
        plt.xlim(xmin=0)
        plt.legend(prop={'size': 6})
        plt.xlabel('Bar width  (mm)')
        plt.ylabel('MTF')
        if PDF_obj is None:
            plt.savefig(folder + 'MTF_vs_width_all_test_patterns.png')
        else:
            PDF_obj.savefig()
        plt.close()

    if False:
        #    Final plot with connected points
        fig = plt.figure(figsize=(10, 6))
        plt.title(folder)
        for i,f_arr in enumerate(f_arrs):
            MTF_arr = MTFs[i]
            MTF_arr_err = MTF_errs[i]
            plot_ind = MTF_arr > -0.01
            w_arr = 1.0 / (2 * f_arrs[i])
            full_basename = img_files[i].split('.')[-2].split('\\')[-1]
            plt.plot(w_arr[plot_ind], MTF_arr[plot_ind],'-x',label=full_basename)

        plt.ylim([0,1])
        plt.xlim(xmin=0)
        plt.legend(prop={'size': 6})
        plt.xlabel('Bar width  (mm)')
        plt.ylabel('MTF')
        if PDF_obj is None:
            plt.savefig(folder + 'MTF_vs_width_all_test_patterns_no_err.png')
        else:
            PDF_obj.savefig()
        plt.close()


    # Put everythign in a Pandas dataframe to make grouping easy
    df = pd.DataFrame(  {'MTF':np.concatenate(MTFs),'MTF_err':np.concatenate(MTF_errs),'f':np.concatenate(f_arrs)} )
    df['bar_width'] = 0.5/df.f
    f_ave   = df.groupby('f')['f'].agg(np.mean).values
    MTF_err_ave = df.groupby('f')['MTF_err'].agg(np.mean).values
    MTF_ave = df.groupby('f')['MTF'].agg(np.mean).values


    if False:
        plt.title(folder)
        plot_ind = MTF_ave > 0.01
        plt.errorbar(f_ave[plot_ind], MTF_ave[plot_ind], yerr=MTF_err_ave[plot_ind],
                     fmt='o', label='average of all images')
        #if f_diff_cutoff != None: plt.plot(f_model,MTF_model,'-',label='model')
        plt.legend()
        plt.xlabel('Frequency  (cy/mm)')
        plt.ylabel('MTF')
        plt.ylim([0, 1])
        plt.xlim(xmin=0)

        if PDF_obj is None:
            plt.savefig(folder + 'MTF_vs_f_average.png')
        else:
            PDF_obj.savefig()
        plt.close()



    if PDF_obj is not None:
        PDF_obj.close()

    print('  writing all plots and output files')
    df.to_csv(folder + 'all_results.csv')

    return()



if __name__== "__main__" :
    folder = "test_folder/"
    file_str = '*.png'
    process_whole_lat_res_folder(folder, file_str,PDF_obj=True)




