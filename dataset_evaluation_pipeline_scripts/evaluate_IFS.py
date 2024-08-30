import os
import time
import csv
import matplotlib.pyplot as plt
import numpy as np
import cv2
from scipy.stats import ks_2samp
import json
import math
from ghostmmaps import process_ghostmaps
from noise_wavelets import get_noise_map
from resammpling import calculate_probability_map_3x3
from MMFusion_AI.analyse_images import process_images
from DFAD2023.predictions import process_images_DFAD


def drop_edge_maps(outputmaps):
    n = outputmaps.shape[2]
    edge = n // 4
    filtered_outputmaps = outputmaps[:,:,edge:-edge]
    return filtered_outputmaps, edge

def save_ghostmaps(original, outputmaps):
    n = outputmaps.shape[2]
    sp = math.ceil(math.sqrt(n+1))
      
    originalBGR = cv2.imread(original)
    #change BGR to RGB for matplotlib
    originalRGB = cv2.cvtColor(originalBGR, cv2.COLOR_BGR2RGB) 
    originalRGB_normalized = originalRGB.astype(np.float32) / 255.0

    plt.subplot(sp, sp, 1)
    plt.imshow(originalRGB_normalized)
    plt.title('Original Image')
    plt.axis('off')
    #add maps:
    for c in range(n):
        plt.subplot(sp, sp, c+2)
        plt.imshow(outputmaps[:, :, c], cmap='gray', vmin=0, vmax=1)
        plt.axis('off')
        plt.title(str(Q_min + c * Q_step))
        plt.draw()

    #main title
    plt.suptitle('Ghost plots for grid offset X = '+ str(shift_x)+" and Y = " + str(shift_y))  

    ghostmap_filepath = os.path.join('ghostmaps_results/IFS', os.path.basename(original) + '_ghostmaps.png')
    plt.savefig(ghostmap_filepath)



#add guarding code block to ensure this is only excecuted when script is called directly, and not when imported by others processes=
if __name__ == '__main__':
    #keep time
    start_time = time.time()
    
    #create ghostmap result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_ghostmaps_IFS.json', 'w') as f:
        json.dump({}, f)
    #create ghostmap output map
    os.makedirs('ghostmaps_results/IFS', exist_ok=True)

    #create probmap 3x3 result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_probability_3x3_IFS.json', 'w') as f:
        json.dump({}, f)
    #create probmap segmented 3x3 output map
    os.makedirs('probability_map_3x3_results/IFS', exist_ok=True)

    #create noise result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_noise_IFS.json', 'w') as f:
        json.dump({}, f)
    #create noise output map
    os.makedirs('noisemap_results/IFS', exist_ok=True)

    #create MM-fusion result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_MM-fusion_IFS.json', 'w') as f:
        json.dump({}, f)
    #create noise output map
    os.makedirs('MM-fusion_results/IFS', exist_ok=True)

    #create DFAD2023 result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_DFAD2023_IFS.json', 'w') as f:
        json.dump({}, f)
    #create noise output map


    #create error log json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('error_log_IFS.json', 'w') as f:
        json.dump('hopefully this remains empty', f)
        f.write('\n')


    #IFS dataset contains pristine and manipulated images
    ifs_pristine_dir = "IFS/pristine"
    ifs_fake_dir = "IFS/fake"

    original_images = [os.path.join(ifs_pristine_dir,path) for path in os.listdir(ifs_pristine_dir)]
    manipulated_images_map = [os.path.join(ifs_fake_dir,path) for path in os.listdir(ifs_fake_dir)]
    manipulated_images = []
    mask_paths = []
    for image_path in manipulated_images_map:
        if "mask" in image_path:
            mask_paths.append(image_path)
        else:
            manipulated_images.append(image_path)

    
    #ghostmaps ----------------------------------------------------------------------------
    print("start calculating ghostmaps")
    #paras
    Q_min = 60
    Q_max = 100
    Q_step = 5
    shift_x = 0
    shift_y = 0
    #number of ghostmaps
    n_ghostmaps = int((Q_max-Q_min)/Q_step)+1
    try:
        #original ---------------------------------
        for image_path in original_images:
            
            outputmaps = process_ghostmaps(image_path, Q_min, Q_max, Q_step, shift_x, shift_y)
            #save total outputmap as image, including original:
            save_ghostmaps(image_path, outputmaps)
            #drop edge cases from ghost map output, so statistical difference is calculated only for non-edge cases:
            outputmaps_filtered, edge = drop_edge_maps(outputmaps)
            #calculate results ghostmaps
            results_stat_difference = []
            #construct mask
            mask = np.zeros((outputmaps_filtered.shape[0], outputmaps_filtered.shape[1]))
            center_x, center_y = mask.shape[0] // 2, mask.shape[1] // 2
            mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255
            for i in range(n_ghostmaps-2*edge):
                outmap = outputmaps_filtered[:,:, i]
                #list values inside mask and values outside mask
                values_inside_mask = outmap[mask > 0.5]
                values_outside_mask = outmap[mask <= 0.5]
                
                #calculate statistical difference using Kolmogorov-Smirnov statistic
                k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
                #calculate medians inside and outside mask
                median_mask = np.nanmedian(values_inside_mask)
                median_outside = np.nanmedian(values_outside_mask)
                results_stat_difference.append([i + edge, k_stat, median_mask, median_outside])

            largest_med_diff = 0
            largest_med_diff_mapnr = -1
            med_inside_mask = 0
            med_outside_mask = 0
            largest_k_diff = 0
            largest_k_diff_mapnr = -1
            for result in results_stat_difference:
                med_diff = abs(result[2] - result[3])
                if largest_med_diff < med_diff:
                    med_inside_mask = result[2]
                    med_outside_mask = result[3]
                    largest_med_diff = med_diff
                    largest_med_diff_mapnr = result[0]
                if largest_k_diff < result[1]:
                    largest_k_diff = result[1]
                    largest_k_diff_mapnr = result[0]        

            #save data in file
            result_data = {
                "image name": image_path,
                "image outputmaps name": os.path.join('ghostmaps_results/IFS', os.path.basename(image_path) + '_ghostmaps.png'),
                "largest med diff map": (Q_min + largest_med_diff_mapnr * Q_step),
                "median inside mask": med_inside_mask,
                "median outside mask": med_outside_mask,
                "absolute median difference": largest_med_diff,
                "map largest k stat": (Q_min + largest_k_diff_mapnr * Q_step),
                "largest k stat diff": largest_k_diff,
                "type": "Original"
            }
            
            with open('results_ghostmaps_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

        #manipulated------------------------------------
        mask_counter = 0
        for image_path in manipulated_images:

            mask_path = mask_paths[mask_counter]
            #ghostmap analysis for manipulated image
            outputmaps = process_ghostmaps(image_path, Q_min, Q_max, Q_step, shift_x, shift_y)
            #save total outputmap as image, including original:
            save_ghostmaps(image_path, outputmaps)
            #drop edge cases from ghost map output, so statistical difference is calculated only for non-edge cases:
            outputmaps_filtered, edge = drop_edge_maps(outputmaps)

            results_stat_difference = []
            n_outputmap_filtered = 0
            #resize mask to outputmap size
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, (outputmaps_filtered.shape[1], outputmaps_filtered.shape[0]), interpolation=cv2.INTER_AREA)
            #make at least top and bottom row = 0
            for i in range(n_ghostmaps-2*edge):
                outmap = outputmaps_filtered[:,:, i]
                #list values inside mask and values outside mask
                values_inside_mask = outmap[resized_mask > 0.5]
                if  len(values_inside_mask) == 0: #it can occur that the mask resizing deletes the mask area, in that= the entire mask becomes black or white; in such cases make top and bottom row different and log to error
                    resized_mask[0,:]=1
                    resized_mask[0,:]=1
                    values_inside_mask = outmap[resized_mask > 0.5]
                    error_info = {
                        "error": "problem with mask for ghostmaps:",
                        "current image": image_path,
                    }
                    with open('error_log_IFS.json', 'a') as f:
                        json.dump(error_info, f)
                        f.write('\n')
                    
                values_outside_mask = outmap[resized_mask <= 0.5]
                if  len(values_outside_mask) == 0:
                    resized_mask[0,:]=0
                    resized_mask[0,:]=0
                    values_inside_mask = outmap[resized_mask > 0.5]
                    values_outside_mask = outmap[resized_mask <= 0.5]
                    error_info = {
                        "error": "problem with mask for ghostmaps:",
                        "current image": image_path,
                    }
                    with open('error_log_IFS.json', 'a') as f:
                        json.dump(error_info, f)
                        f.write('\n')
                
                #calculate statistical difference using Kolmogorov-Smirnov statistic
                k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
                #calculate medians inside and outside mask
                median_mask = np.nanmedian(values_inside_mask)
                median_outside = np.nanmedian(values_outside_mask)
                results_stat_difference.append([n_outputmap_filtered + edge, k_stat, median_mask, median_outside])
                n_outputmap_filtered +=1

            largest_med_diff = 0
            largest_med_diff_mapnr = -1
            med_inside_mask = 0
            med_outside_mask = 0
            largest_k_diff = 0
            largest_k_diff_mapnr = -1
            for result in results_stat_difference:
                med_diff = abs(result[2] - result[3])
                if largest_med_diff < med_diff:
                    med_inside_mask = result[2]
                    med_outside_mask = result[3]
                    largest_med_diff = med_diff
                    largest_med_diff_mapnr = result[0]
                if largest_k_diff < result[1]:
                    largest_k_diff = result[1]
                    largest_k_diff_mapnr = result[0]        

            #save data in file
            result_data = {
                "image name": image_path,
                "image outputmaps name": os.path.join('ghostmaps_results/IFS', os.path.basename(image_path) + '_ghostmaps.png'),
                "largest med diff map": (Q_min + largest_med_diff_mapnr * Q_step),
                "median inside mask": med_inside_mask,
                "median outside mask": med_outside_mask,
                "absolute median difference": largest_med_diff,
                "map largest k stat": (Q_min + largest_k_diff_mapnr * Q_step),
                "largest k stat diff": largest_k_diff,
                "type": "manipulated"
            }
            
            with open('results_ghostmaps_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            
            mask_counter +=1

        #calculate expired time and write to error log
        end_time = time.time()
        time_elapsed_info = {
                "Time elapsed ghost maps": str(end_time - start_time)
            }
        with open('error_log_IFS.json', 'a') as f:
            json.dump(time_elapsed_info, f)
            f.write('\n')
        start_time = end_time

    except Exception as e:
        error_info = {
            "error": "Error occurred in map, for ghostmaps:",
            "current image": image_path,
            "exception": str(e)
        }
        with open('error_log_IFS.json', 'a') as f:
            json.dump(error_info, f)
            f.write('\n')

    #probability maps-----------------------------------------------------------------------------------------------
    #construct mask probability map 3x3     
    print("start calculating probabilitymaps")   

    #original images ---------------------------
    for image_path in original_images:

        try:
            #probabilitymap 3x3 + save --------------------------------------------------------------------------
            probability_map_3x3 = calculate_probability_map_3x3(image_path)
            probmap_filepath = os.path.join('probability_map_3x3_results/IFS', os.path.basename(image_path) + '_probmap.png')
            probability_map_3x3 = probability_map_3x3 *255 #probmap_filepath = normalized between 0 and 1, so rescale to 0 - 255
            cv2.imwrite(probmap_filepath, probability_map_3x3)#save map

            #construct mask
            mask = np.zeros(probability_map_3x3.shape)
            center_x, center_y = probability_map_3x3.shape[0] // 2, probability_map_3x3.shape[1] // 2
            mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255

            #results prob map
            values_inside_mask = probability_map_3x3[mask > 0.5]
            values_outside_mask = probability_map_3x3[mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image_path,
                "image prob name": probmap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "original"
            }

            with open('results_probability_3x3_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

        except Exception as e:
            error_info = {
                "error": "Error occurred in file, for prob maps:",
                "current image": image_path,
                "exception": str(e)
            }
            with open('error_log_IFS.json', 'a') as f:
                json.dump(error_info, f)
                f.write('\n')


    #manipulated images ----------------------
    mask_counter = 0
    for image_path in manipulated_images:
        try:
            mask_path = mask_paths[mask_counter]        
        
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            resized_mask = mask[1:mask.shape[0]-1,1:mask.shape[1]-1]


            #probabilitymap 3x3 + save --------------------------------------------------------------------------
            probability_map_3x3 = calculate_probability_map_3x3(image_path)
            probmap_filepath = os.path.join('probability_map_3x3_results/IFS', os.path.basename(image_path) + '_probmap.png')
            probability_map_3x3 = probability_map_3x3 *255 #probmap_filepath = normalized between 0 and 1, so rescale to 0 - 255
            cv2.imwrite(probmap_filepath, probability_map_3x3)#save map

            #resize mask if dimentions differ from prob map
            if resized_mask.shape != probability_map_3x3.shape:
                resized_mask = cv2.resize(resized_mask, (probability_map_3x3.shape[1], probability_map_3x3.shape[0]), interpolation=cv2.INTER_NEAREST)

            #results prob map
            values_inside_mask = probability_map_3x3[resized_mask > 0.5]
            values_outside_mask = probability_map_3x3[resized_mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image_path,
                "image prob name": probmap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "manipulated"
            }

            with open('results_probability_3x3_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

        except Exception as e:
            error_info = {
                "error": "Error occurred in file, for prob maps:",
                "current image": image_path,
                "exception": str(e)
            }
            with open('error_log_IFS.json', 'a') as f:
                json.dump(error_info, f)
                f.write('\n')    
        
        mask_counter +=1

    #calculate expired time and write to error log
    end_time = time.time()
    time_elapsed_info = {
            "Time elapsed probability maps": str(end_time - start_time)
        }
    with open('error_log_IFS.json', 'a') as f:
        json.dump(time_elapsed_info, f)
        f.write('\n')
    start_time = end_time

    #noise maps --------------------------------------------------------------------------------------------------
    print("start calculating noise maps")

    try:
        #original images ---------------------------
        for image_path in original_images:
            noise_map = get_noise_map(image_path)
            noisemap_filepath = os.path.join('noisemap_results/IFS', os.path.basename(image_path) + 'noisemap.png')
            cv2.imwrite(noisemap_filepath, noise_map)#save map

            #second map
            originalBGR = cv2.imread(image_path)
            #change BGR to RGB for matplotlib
            originalRGB = cv2.cvtColor(originalBGR, cv2.COLOR_BGR2RGB) 
            originalRGB_normalized = originalRGB.astype(np.float32) / 255.0
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(originalRGB_normalized)
            plt.title('Original Image')
            plt.axis('off')
            #add map:
            plt.subplot(1,2,2)
            plt.imshow(noise_map, cmap='gray')
            plt.axis('off')
            plt.title("noise map")
            plt.draw()

            noisemap_2_filepath = os.path.join('noisemap_results/IFS', os.path.basename(image_path) + '_noisemap_2.png')
            plt.savefig(noisemap_2_filepath)

            #construct mask noise original
            mask = np.zeros(noise_map.shape)
            center_x, center_y = noise_map.shape[0] // 2, noise_map.shape[1] // 2
            mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255

            values_inside_mask = noise_map[mask > 0.5]
            values_outside_mask = noise_map[mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image_path,
                "image noise name": noisemap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "original"
            }
            
            with open('results_noise_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

        #manipulated images -----------------------------------
        mask_counter = 0
        for image_path in manipulated_images:
            mask_path = mask_paths[mask_counter]   

            noise_map = get_noise_map(image_path)
            noisemap_filepath = os.path.join('noisemap_results/IFS', os.path.basename(image_path) + 'noisemap.png')
            cv2.imwrite(noisemap_filepath, noise_map)#save map

            #second map
            originalBGR = cv2.imread(image_path)
            #change BGR to RGB for matplotlib
            originalRGB = cv2.cvtColor(originalBGR, cv2.COLOR_BGR2RGB) 
            originalRGB_normalized = originalRGB.astype(np.float32) / 255.0
            plt.clf()
            plt.subplot(1,2,1)
            plt.imshow(originalRGB_normalized)
            plt.title('Original Image')
            plt.axis('off')
            #add map:
            plt.subplot(1,2,2)
            plt.imshow(noise_map, cmap='gray')
            plt.axis('off')
            plt.title("noise map")
            plt.draw()

            noisemap_2_filepath = os.path.join('noisemap_results/IFS', os.path.basename(image_path) + '_noisemap_2.png')
            plt.savefig(noisemap_2_filepath)

            #construct mask noise manipulated
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, (noise_map.shape[1], noise_map.shape[0]), interpolation=cv2.INTER_AREA)

            #list values inside mask and values outside mask
            values_inside_mask = noise_map[resized_mask > 0.5]
            if  len(values_inside_mask) == 0:
                resized_mask[0,:]=1
                resized_mask[0,:]=1
                values_inside_mask = noise_map[resized_mask > 0.5]
                error_info = {
                    "error": "problem with mask for noisemaps:",
                    "current image": image_path,
                }
                with open('error_log_IFS.json', 'a') as f:
                    json.dump(error_info, f)
                    f.write('\n')
                
            values_outside_mask = noise_map[resized_mask <= 0.5]
            if  len(values_outside_mask) == 0:
                resized_mask[0,:]=0
                resized_mask[0,:]=0
                values_inside_mask = noise_map[resized_mask > 0.5]
                values_outside_mask = noise_map[resized_mask <= 0.5]
                error_info = {
                    "error": "problem with mask for noisemaps:",
                    "current image": image_path,
                }
                with open('error_log_IFS.json', 'a') as f:
                    json.dump(error_info, f)
                    f.write('\n')

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image_path,
                "image noise name": noisemap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "manipulated"
            }
            
            
            with open('results_noise_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

            mask_counter += 1

    except Exception as e:
        error_info = {
            "error": "Error occurred for noisemaps:",
            "current image": image_path,
            "exception": str(e)
        }
        with open('error_log_IFS.json', 'a') as f:
            json.dump(error_info, f)
            f.write('\n')

   #calculate expired time and write to error log
    end_time = time.time()
    time_elapsed_info = {
            "Time elapsed noise maps": str(end_time - start_time)
        }
    with open('error_log_IFS.json', 'a') as f:
        json.dump(time_elapsed_info, f)
        f.write('\n')
    start_time = end_time


    #AI predictions --------------------------------------------------------------
    #mm-fusion
    #mm-fusion expects images as absolute paths
    print("start MM-Fusion")
    try:
        #calculate all prediction maps
        list_image_paths = original_images + manipulated_images
        base_path = os.getcwd()
        list_absolute_image_paths = [os.path.join(base_path, relative_path) for relative_path in list_image_paths]
        detection_scores, map_locations = process_images(list_absolute_image_paths, 'MM-fusion_results/IFS/')

        #calculate detection original images ---------------------------------------------
        mm_counter = 0
        for image in original_images:
            #map mmfusion
            fusion_map = cv2.imread(map_locations[mm_counter], cv2.IMREAD_GRAYSCALE)

            #orig mask contruction
            mask = np.zeros(fusion_map.shape)
            center_x, center_y = fusion_map.shape[0] // 2, fusion_map.shape[1] // 2
            mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255

            #results prob map
            values_inside_mask = fusion_map[mask > 0.5]
            values_outside_mask = fusion_map[mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image,
                "image output map location": map_locations[mm_counter],
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "detection score AI-model": detection_scores[mm_counter],
                "type": "original"
            }
            
            with open('results_MM-fusion_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            mm_counter += 1

        #calculate detections based on maps
        mask_counter = 0
        for image in manipulated_images:
            mask_path = mask_paths[mask_counter]
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            #map mmfusion
            fusion_map = cv2.imread(map_locations[mm_counter],cv2.IMREAD_GRAYSCALE)


            #resize mask if dimentions exceed fusion map
            if mask.shape != fusion_map.shape:
                mask = cv2.resize(mask, (fusion_map.shape[1], fusion_map.shape[0]), interpolation=cv2.INTER_NEAREST)
            
            #results prob map
            values_inside_mask = fusion_map[mask > 0.5]
            values_outside_mask = fusion_map[mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": image,
                "image output map location": map_locations[mm_counter],
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "detection score AI-model": detection_scores[mm_counter],
                "type": "manipulated"
            }
            
            with open('results_MM-fusion_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            mm_counter += 1
            mask_counter +=1
        
    except Exception as e:
        error_info = {
            "error": "Error occurred in map, for MM-fusion:",
            "exception": str(e)
        }
        with open('error_log_IFS.json', 'a') as f:
            json.dump(error_info, f)
            f.write('\n')

    #calculate expired time and write to error log
    end_time = time.time()
    time_elapsed_info = {
            "Time elapsed MM-fusion maps": str(end_time - start_time)
        }
    with open('error_log_IFS.json', 'a') as f:
        json.dump(time_elapsed_info, f)
        f.write('\n')
    start_time = end_time

    #DFAD ------------------------------------------------
    print("start DFAD2023 predictions")
    try:
        list_image_paths = original_images + manipulated_images
        detection_scores = process_images_DFAD(list_image_paths)
        #all ------------------------------------------------------------------------------------
        mm_counter = 0
        for image in original_images:
            #save data in file
            result_data = {
                "image_name": image,
                "prediction score AI-model": detection_scores[mm_counter].item(),
                "type": "original"
            }

            with open('results_DFAD2023_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            mm_counter += 1

        for image in manipulated_images:
            #save data in file
            result_data = {
                "image_name": image,
                "prediction score AI-model": detection_scores[mm_counter].item(),
                "type": "manipulated"
            }

            with open('results_DFAD2023_IFS.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            mm_counter += 1
        
    except Exception as e:
        error_info = {
            "error": "Error occurred in map, for DFAD2023:",
            "exception": str(e)
        }
        with open('error_log_IFS.json', 'a') as f:
            json.dump(error_info, f)
            f.write('\n')
    
    #calculate expired time and write to error log
    end_time = time.time()
    time_elapsed_info = {
            "Time elapsed DFAD2023 prediction": str(end_time - start_time)
        }
    with open('error_log_IFS.json', 'a') as f:
        json.dump(time_elapsed_info, f)
        f.write('\n')
    start_time = end_time
    

    
