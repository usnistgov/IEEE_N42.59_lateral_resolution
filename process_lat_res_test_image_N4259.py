

import os
import matplotlib.pyplot as plt
import numpy as np
from scipy.ndimage.filters import gaussian_filter
import cv2



def get_Fourier_profile(bar_img):
    n_rows = bar_img.shape[0]
    n_cols = bar_img.shape[1]
    f_grid = np.arange(0, int(round(n_cols/2)) ) / n_cols

    fourier_img = np.zeros(  (n_rows,n_cols)  )

    for i in np.arange(bar_img.shape[0]):
        bar_profile = bar_img[i,:]

        profile_FFT = np.fft.fft(bar_profile)
        #plt.plot(np.abs(profile_FFT)/ N)
        #plt.show()
        profile_FFT = np.abs(profile_FFT) / n_cols

        fourier_img[i,:] = profile_FFT
        #f_grid =  np.fft.fftfreq(theta_n_sample, theta_arr[1 ] -theta_arr[0])[0: theta_n_sample /2]






    profile_fft_mean   = np.mean(fourier_img, axis=0)
    profile_fft_mean_err = np.std(fourier_img, axis=0)/np.sqrt(n_rows)

    if False:
        plt.close()
        print(profile_fft_std[0:30])
        print('profile_fft_std[10]',profile_fft_std[10])
        print('% err', profile_fft_std[10]/profile_fft_mean[10])
        for i in range(n_rows):
            plt.plot(fourier_img[i,0:30],'x')
        plt.show()

    return(profile_fft_mean_err,profile_fft_mean,f_grid)





def get_pattern_info(mm_per_px):

    def generate_pattern_struct(w):
        pad = 30.0
        fid_diam_mm = 10.0
        n_slots       = 10
        bar_height    = 3*w
        if w > 12: n_slots = 5
        if w < 6.5: bar_height = 20
        if w < 3.5: bar_height = 15
        if w < 2.5: bar_height = 10

        struct = {
            'n_slots':n_slots,
            'pattern_width_mm':((2*n_slots-2)*w),
            'pattern_width_px':((2*n_slots-2)*w)/mm_per_px[1],
            'padding_px': pad/ mm_per_px[1],
            'w_in_mm':w,
            'w_in_px':(w / mm_per_px[1]),
            'fiducial_sep_mm':(w*(2*n_slots - 1) + 2*pad),
            'fiducial_sep_px': (w*(2*n_slots - 1) + 2*pad)/mm_per_px[1],
            'fid_diam_px': (fid_diam_mm/mm_per_px[1]),
            'bar_height_in_px': (bar_height/mm_per_px[0]),
            'ROI_height_in_px':(bar_height /(3*mm_per_px[0])),
        }
        return(struct)

    info = {
        20:generate_pattern_struct(20),
        15:generate_pattern_struct(15),
        10: generate_pattern_struct(10),
        8: generate_pattern_struct(8),
        7: generate_pattern_struct(7),
        6: generate_pattern_struct(6),
        5:generate_pattern_struct(5),
        4: generate_pattern_struct(4),
        3: generate_pattern_struct(3),
        2: generate_pattern_struct(2),
        1.5: generate_pattern_struct(1.5),
        1: generate_pattern_struct(1),
        0.5: generate_pattern_struct(0.5),
    }

    return(info)


def crop_to_large_img(full_img,i_row,pattern_info):
    half_vert_width = int(np.round(pattern_info['bar_height_in_px']*3/2))
    if half_vert_width < 1: half_vert_width = 1

    x_c = int(round(full_img.shape[1]/2))
    half_width = int(round( 1.3*pattern_info['fiducial_sep_px']/2  ))+1

    large_pattern_img = full_img.copy()

    vert_min = np.max(   (0,round(i_row-half_vert_width))      )
    vert_max =  np.min(   (round(i_row+half_vert_width),full_img.shape[0]-1)    )
    horiz_min = np.max(  (0,int(round(x_c - half_width)))           )
    horiz_max = np.min(  (int(round(x_c + half_width)),full_img.shape[1]-1)           )

    large_pattern_img = large_pattern_img[vert_min:vert_max,horiz_min:horiz_max]

    if False:
        plt.subplot(2,1,1)
        plt.imshow(full_img)
        plt.subplot(2,1,2)
        plt.imshow(large_pattern_img)
        plt.show()

    return(large_pattern_img)


# large_cropped_img is 3 slots higi


def crop_and_normalize_using_fiducials(large_cropped_img,pattern_info,plot=True,plot_folder='',PDF_obj=None,filename=''):
    fid_img    = gaussian_filter(large_cropped_img, sigma=pattern_info['fid_diam_px']/2,mode='nearest')
    lower_pass = gaussian_filter(large_cropped_img, sigma=pattern_info['fid_diam_px']*2,mode='nearest')

    vert_edge_pad = 0.2
    vert_edge_pad = 0.35
    horiz_edge_pad = 0.08

    # high pass filter
    fid_img = fid_img*1.0 - lower_pass


    bckg    = np.max(fid_img)
    ww = fid_img.shape[1]  # width
    hh = fid_img.shape[0]  # height

    sh = large_cropped_img.shape
    fid_img_left = fid_img.copy()
    i_boundary = int(( 0.5*(ww - pattern_info['pattern_width_px']*1.15 )))-1

    # to find the left fiducial, we have to find any areas that could be darker than it
    # and set them equal to bckg (a bright grayscale value)
    fid_img_left[:,i_boundary:] = bckg # everything right of the fiducial
    fid_img_left[:, 0:int(sh[1]*horiz_edge_pad)] = bckg  # hoiz edge
    fid_img_left[int(sh[0]*(1-vert_edge_pad)):,:] = bckg # the top
    fid_img_left[:int(sh[0]*vert_edge_pad),:] = bckg # the bottom
    # for subpixel resolution, I scaled up the image by a factor of 2
    fid_img_left = cv2.resize(fid_img_left, None, fx=2, fy=2,interpolation=cv2.INTER_LINEAR)
    fid_left = np.unravel_index( np.argmin(fid_img_left) , fid_img_left.shape )
    # undo the scaling
    fid_left = np.array(fid_left)/2.0


    fid_img_right = fid_img.copy()
    fid_img_right[:,0:(ww - i_boundary)+1] = bckg
    fid_img_right[:, int(sh[1]*(1-horiz_edge_pad)):] = bckg  # hoiz edge
    fid_img_right[int(sh[0]*(1-vert_edge_pad)):,:] = bckg
    fid_img_right[:int(sh[0]*vert_edge_pad),:] = bckg
    fid_img_right = cv2.resize(fid_img_right, None, fx=2, fy=2)
    fid_right = np.unravel_index( np.argmin(fid_img_right) , fid_img_right.shape )
    fid_right = np.array(fid_right)/ 2.0

    # now crop down to a width of 20*w
    mm_per_px_accurate = pattern_info['fiducial_sep_mm']/(fid_right[1] - fid_left[1] )

    x_c = 0.5*(fid_left[1] + fid_right[1])
    y_c = int(round(0.5*(fid_left[0] + fid_right[0])))

    ROI_width_px =  int(round(pattern_info['pattern_width_mm']/mm_per_px_accurate))
    ROI_ends     = int(round(x_c - ROI_width_px/2)) + np.array([0,ROI_width_px])

    final_height = int(np.ceil(pattern_info['ROI_height_in_px']))
    half_final_height = int(round(pattern_info['ROI_height_in_px'] / 2))
    if final_height < 1: final_height=1

    y_0 = y_c-half_final_height
    if y_0 < 0: y_0=0

    unnormalized_test_pattern = large_cropped_img[y_0:y_0+final_height,ROI_ends[0]:ROI_ends[1]+1]
    norm_height = int(np.ceil(pattern_info['ROI_height_in_px']/2))
    if norm_height < 1:
        norm_height=1

    y_0_upper = int(y_0 -pattern_info['bar_height_in_px'])
    if y_0_upper < 0: y_0_upper = 0
    upper_norm_ROI = large_cropped_img[y_0_upper:y_0_upper+norm_height,ROI_ends[0]:ROI_ends[1]+1]
    y_0_lower = int(y_0 + pattern_info['bar_height_in_px'])
    lower_norm_ROI = large_cropped_img[y_0_lower:y_0_lower+norm_height,ROI_ends[0]:ROI_ends[1]+1]

    if len(unnormalized_test_pattern) < 1:
        print('bad1')
        print(ROI_ends)
        a=1
    if len(lower_norm_ROI) < 1:
        print('bad2')
        print(ROI_ends)
        a = 1
    if len(upper_norm_ROI) < 1:
        print('bad3')
        print(ROI_ends)
        print("pattern_info['bar_height_in_px']",pattern_info['bar_height_in_px'])
        print(y_0_upper,y_0_upper+norm_height,ROI_ends[0],ROI_ends[1]+1)
        a=1

    unnormalized_signal = np.mean(unnormalized_test_pattern, axis=0)
    upper_signal        = np.mean(upper_norm_ROI,axis=0)
    lower_signal        = np.mean(lower_norm_ROI,axis=0)
    normalized_signal       = 2*unnormalized_signal/(upper_signal+lower_signal)

    norm = (upper_signal+lower_signal)/2.0
    normalized_test_pattern = unnormalized_test_pattern.copy()/norm


    def plot_ROI_box(x_0,y_0,width,height,color='red'):
        plt.plot([y_0,y_0],[x_0,x_0+height],'--',color=color)
        plt.plot([y_0+width, y_0+width], [x_0, x_0 + height], '--', color=color)
        plt.plot([y_0,y_0+width],[x_0,x_0],'--',color=color)
        plt.plot([y_0,y_0+width], [x_0+height, x_0+height], '--', color=color)

    if plot and False:
        plt.title('fiducials')
        plt.imshow(fid_img, cmap='gray')
        # plot fiducial positions
        plt.plot(fid_left[1], fid_left[0], 'rx')
        plt.plot(fid_right[1], fid_right[0], 'rx')
        plt.plot(x_c, y_c, 'rx')
        # plot ROI ends
        plot_ROI_box(y_0, ROI_ends[0], ROI_width_px, final_height)
        plot_ROI_box(y_0_upper, ROI_ends[0], ROI_width_px, norm_height, color='green')
        plot_ROI_box(y_0_lower, ROI_ends[0], ROI_width_px, norm_height, color='green')

        path = plot_folder + 'fiducials_and_crops/'
        if not os.path.exists(path): os.makedirs(path)


        if PDF_obj is None:
            plt.savefig(plot_folder + 'fiducials_and_crops/' + str(pattern_info['w_in_mm']) + 'mm_fid_finder.png')
        else:
            PDF_obj.savefig()




    if plot:
        fig = plt.figure(figsize=(10, 6))
        plt.title('ROIs and fiducials ({} mm)'.format(pattern_info['w_in_mm']))
        plt.suptitle(filename)
        plt.imshow(large_cropped_img, cmap='gray')
        # plot fiducial positions
        plt.plot(fid_left[1], fid_left[0], 'rx')
        plt.plot(fid_right[1], fid_right[0], 'rx')
        plt.plot(x_c, y_c, 'rx')
        # plot ROI ends
        plot_ROI_box(y_0, ROI_ends[0], ROI_width_px, final_height)
        plot_ROI_box(y_0_upper, ROI_ends[0], ROI_width_px, norm_height, color='green')
        plot_ROI_box(y_0_lower, ROI_ends[0], ROI_width_px, norm_height, color='green')

        path = plot_folder + 'fiducials_and_crops/'
        if not os.path.exists(path): os.makedirs(path)


        if PDF_obj is None:
            plt.savefig(plot_folder + 'fiducials_and_crops/' + str(pattern_info['w_in_mm']) + 'mm_all_ROIs.png')
        else:
            PDF_obj.savefig()
        plt.close()

    if plot:
        fig = plt.figure(figsize=(10, 6))

        try: img_max = np.nanmax( ( unnormalized_test_pattern.max(), lower_norm_ROI.max() ))
        except: return(normalized_test_pattern)

        plt.subplot(4, 2, 1)
        plt.title('Upper norm ROI ')
        plt.imshow(upper_norm_ROI, vmin=0, vmax=img_max)

        plt.subplot(4, 2, 3)
        plt.title('Unnormalized test pattern')
        plt.imshow(unnormalized_test_pattern, vmin=0, vmax=img_max)

        plt.subplot(4, 2, 5)
        plt.title('Lower norm ROI')
        plt.imshow(lower_norm_ROI, vmin=0, vmax=img_max)

        plt.subplot(4, 2, 7)
        plt.title('Normalized test pattern')
        plt.imshow(normalized_test_pattern)

        plt.subplot(2, 2, 2)
        plt.title('Un-Normalized Data')
        plt.plot(unnormalized_signal*2*200,label='2 * S(x)')
        plt.plot(norm*2*200,label='U(x) + L(x)')
        plt.ylabel('Pixel value')
        plt.xlabel('Position  (px)')
        #plt.legend()

        plt.subplot(2, 2, 4)
        plt.title('Bar Profile Function, B(x)')
        plt.plot(normalized_signal,label='B(x)')
        plt.xlabel('Position  (px)')
        #plt.legend()

        plt.tight_layout()

        path = plot_folder + 'fiducials_and_crops/'
        if not os.path.exists(path): os.makedirs(path)

        if PDF_obj is None:
            plt.savefig(plot_folder + 'fiducials_and_crops/' + str(pattern_info['w_in_mm']) + 'mm_normalization.png')
        else:
            PDF_obj.savefig()
        plt.close()

    return(normalized_test_pattern)





# This calculates the MTF from a single test pattern
def calculate_MTF_from_pattern_img(bar_img,pattern_info,plot=True,plot_folder='',PDF_obj=None):
    profile_fft_mean_err, profile_fft_mean, f_grid = get_Fourier_profile(bar_img)
    w = pattern_info['w_in_mm']

    fig = plt.figure(figsize=(10, 6))

    plt.subplot(2, 2, 1)
    plt.title('Profile function: {0} mm'.format(w), fontsize=12)
    # plt.imshow(bar_img,interpolation='none')
    plt.plot(np.mean(bar_img,axis=0))
    plt.subplot(2, 2, 2)
    plt.plot(f_grid, profile_fft_mean[0:len(f_grid)])


    i_signal = pattern_info['n_slots'] -1 # minus 1 because we dont use the outer edge anynmore
    if i_signal < len(f_grid):
        plt.plot(f_grid[i_signal], profile_fft_mean[i_signal], 'ro')
        MTF = (profile_fft_mean[i_signal]) * np.pi
        MTF_err = profile_fft_mean_err[i_signal] * np.pi   # this is just the variation between rows
        # now calculate the effect of spectral leakage, assuming width uncertainty of 0.5
        MTF_err_leakage = MTF * (1.0 - np.sinc(np.pi*0.5/(2*pattern_info['w_in_px'])))
        MTF_err_leakage = 2.0 / pattern_info['w_in_px']
        MTF_err_leakage = MTF*(1 - np.sinc(np.pi * 0.5 * i_signal/(bar_img.shape[1])) )
        MTF_err = np.sqrt(MTF_err**2 + MTF_err_leakage**2)

        #print('var_err', MTF_err, 'leakage_err', MTF_err_leakage)
    else:
        MTF = 0.0
        MTF_err = 0.0




    plt.errorbar(f_grid, profile_fft_mean[0:len(f_grid)], profile_fft_mean_err[0:len(f_grid)])
    plt.xlabel('Spatial freq.  (cy/px)', fontsize=10)
    plt.xticks(fontsize=10)
    plt.title('MTF = {0:.2f}'.format(MTF))
    plt.ylabel('FFT coeff.', fontsize=10)

    plt.subplot(2, 1, 2)
    plt.imshow(bar_img)

    path = plot_folder+'MTF_output/'
    if not os.path.exists(path): os.makedirs(path)
    filename = plot_folder+'MTF_output/{0}mm_bar_analysis.png'.format(w)
    plt.tight_layout()


    if PDF_obj is None and False:
        plt.savefig(filename)
    else:
        PDF_obj.savefig()
    plt.close()

    return( (MTF,MTF_err) )

def create_MTF_model(f_max, aureole=(0.0,10.0), f_diff_cutoff=0.25):
    f_model = np.linspace(0,f_max,num=100)

    MTF_diff_limited = 1.0 - f_model / (f_diff_cutoff)
    MTF_diff_limited[MTF_diff_limited < 0] = 0.0

    amp = aureole[0]
    sigma = aureole[1]
    MTF_aureole = ((1-amp) + amp * np.exp(-0.5 * (sigma * f_model) ** 2))


    MTF_model = MTF_diff_limited * MTF_aureole

    return( f_model, MTF_model )



def process_lat_res_test_image(img_filename,info_csv,plot_folder='',f_diff_cutoff=None,aureole=(0.0,50),PDF_obj=None):
    full_img = cv2.imread(img_filename,-1)
    # if it's a 3 channel img,  just use one channel
    if len(full_img.shape) == 3: full_img = full_img[:, :, 1]


    w_arr = info_csv.w.values
    i_row = (info_csv.row_height.values * full_img.shape[0] ).astype(int)

    MTF_arr = w_arr * 0.0
    MTF_arr_err = MTF_arr.copy()

    mm_per_px = np.array((350, 350)) / full_img.shape # this is just a rough initial estimate representing
    # the size of the test object 350 mm by 350 mm
    # precise local length scales come from the fidicials
    pattern_info = get_pattern_info(mm_per_px)

    print('        slot pattern:', end='')
    for i, w in enumerate(w_arr):
        print(" w={0},".format(w), end='')
        # crop down from full image to an image with just the pattern, fiducials and norm strips
        large_bar_img = crop_to_large_img(full_img, i_row[i], pattern_info[w])

        # find the fidicials, then crop more precisely
        bar_img = crop_and_normalize_using_fiducials(large_bar_img, pattern_info[w], plot_folder=plot_folder,PDF_obj=PDF_obj,filename=img_filename)

        MTF_results = calculate_MTF_from_pattern_img(bar_img, pattern_info[w], plot_folder=plot_folder,PDF_obj=PDF_obj)
        MTF_arr[i] = MTF_results[0]
        MTF_arr_err[i] = MTF_results[1]
    print('')

    f_arr = 1.0 / (2 * w_arr)

    if False:
        plt.plot(w_arr, MTF_arr, 'ro-')
        plt.xlabel('Bar width  (mm)')
        plt.ylabel('MTF')

        if PDF_obj is None:
            plt.savefig(plot_folder + 'MTF_vs_w.png')
            plt.close()
        else:
            PDF_obj.savefig()



        plot_ind = MTF_arr > -0.01
        plt.errorbar(f_arr[plot_ind], MTF_arr[plot_ind], MTF_arr_err[plot_ind], fmt='ro')
        if f_diff_cutoff != None:
            f_model,MTF_model = create_MTF_model(0.25,f_diff_cutoff=f_diff_cutoff,aureole=aureole)
            plt.plot(f_model, MTF_model)
        plt.xlabel('Spatial Frequency  (cy/mm)')
        plt.ylabel('MTF')

        plt.ylim((0,1))

        if PDF_obj is None:
            plt.savefig(plot_folder + 'MTF_vs_f.png')
            plt.close()
        else:
            PDF_obj.savefig()

    return( (f_arr,MTF_arr,MTF_arr_err) )















