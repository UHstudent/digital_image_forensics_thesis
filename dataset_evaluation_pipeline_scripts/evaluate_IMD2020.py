import os
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

def get_images_map(submap):
    orig_image = []
    manipulated_images = []
    masks = []
    images_paths = [os.path.join(submap, f) for f in os.listdir(submap) if "_" in f] #only include images containing '_'
    for im in images_paths:
        if 'orig' in im:
            orig_image.append(im)
        elif 'mask' in im:
            masks.append(im)
        else:
            manipulated_images.append(im)
    return orig_image, manipulated_images, masks

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
    plt.clf()
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

    ghostmap_filepath = os.path.join('ghostmaps_results/IMD2020', os.path.basename(original) + '_ghostmaps.png')
    plt.savefig(ghostmap_filepath)


#add guarding code block to ensure this is only excecuted when script is called directly, and not when imported by others processes=
if __name__ == '__main__':

    #create list submaps of /IMD2020
    submaps_IMD2020 = [os.path.join('IMD2020', d) for d in os.listdir('IMD2020') if os.path.isdir(os.path.join('IMD2020', d))]


    #create ghostmap result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_ghostmaps_IMD2020.json', 'w') as f:
        json.dump({}, f)
    #create ghostmap output map
    os.makedirs('ghostmaps_results/IMD2020', exist_ok=True)

    #create probmap 3x3 result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_probability_3x3_IMD2020.json', 'w') as f:
        json.dump({}, f)
    #create probmap segmented 3x3 output map
    os.makedirs('probability_map_3x3_results/IMD2020', exist_ok=True)

    #create noise result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_noise_IMD2020.json', 'w') as f:
        json.dump({}, f)
    #create noise output map
    os.makedirs('noisemap_results/IMD2020', exist_ok=True)

    #create MM-fusion result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_MM-fusion_IMD2020.json', 'w') as f:
        json.dump({}, f)
    #create noise output map
    os.makedirs('MM-fusion_results/IMD2020', exist_ok=True)

    #create DFAD2023 result json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('results_DFAD2023_IMD2020.json', 'w') as f:
        json.dump({}, f)
    #create noise output map


    #create error log json file, so we can store results in file each time, preserving results in case of unexpected interupt
    with open('error_log.json', 'w') as f:
        json.dump('hopefully this remains empty', f)

    #for each submap, take originals, manipulated, masks
    for submap in submaps_IMD2020:
        print(submap)
        orig_images, manipulated_images, masks = get_images_map(submap)

        #ghostmap results
        Q_min = 60
        Q_max = 100
        Q_step = 5
        shift_x = 0
        shift_y = 0
        #number of ghostmaps
        n_ghostmaps = int((Q_max-Q_min)/Q_step)+1
        #ghostmap analysis for original image:
        for orig in orig_images: 
            outputmaps = process_ghostmaps(orig, Q_min, Q_max, Q_step, shift_x, shift_y)

            #ghostmaps
            #save total outputmap as image, including original:
            save_ghostmaps(orig, outputmaps)
            #drop edge cases from ghost map output, so statistical difference is calculated only for non-edge cases:
            outputmaps_filtered, edge = drop_edge_maps(outputmaps)
            #calculate results ghostmaps
            results_stat_difference = []
            for i in range(n_ghostmaps-2*edge):
                outmap = outputmaps_filtered[:,:, i]
                mask = np.zeros(outmap.shape)
                center_x, center_y = outmap.shape[0] // 2, outmap.shape[1] // 2
                mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255
                
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
                "image_name": orig,
                "image outputmaps name": os.path.join('ghostmaps_results/IMD2020', os.path.basename(orig) + '_ghostmaps.png'),
                "largest med diff map": (Q_min + largest_med_diff_mapnr * Q_step),
                "median inside mask": med_inside_mask,
                "median outside mask": med_outside_mask,
                "absolute median difference": largest_med_diff,
                "map largest k stat": (Q_min + largest_k_diff_mapnr * Q_step),
                "largest k stat diff": largest_k_diff,
                "type": "original"
            }
            
            with open('results_ghostmaps_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')


            #probabilitymap ------------------------------------------------------------------------------------
            #construct mask probability map 3x3
            #construct mask noise original
            original_im = cv2.imread(orig, cv2.IMREAD_GRAYSCALE)
            mask = np.zeros(original_im.shape)
            center_x, center_y = original_im.shape[0] // 2, original_im.shape[1] // 2
            mask[center_x - center_x//2:center_x + center_x//2, center_y - center_y//2:center_y + center_y//2] = 255
            resized_mask = mask[1:mask.shape[0]-1,1:mask.shape[1]-1]

            #probabilitymap 3x3 + save --------------------------------------------------------------------------
            probability_map_3x3 = calculate_probability_map_3x3(orig)
            probmap_filepath = os.path.join('probability_map_3x3_results/IMD2020', os.path.basename(orig) + '_probmap.png')
            probability_map_3x3 = probability_map_3x3 *255 #probmap_filepath = normalized between 0 and 1, so rescale to 0 - 255
            cv2.imwrite(probmap_filepath, probability_map_3x3)#save map

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
                "image_name": orig,
                "image prob name": probmap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "original"
            }
            

            with open('results_probability_3x3_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')



            #noise -------------------------------------------------------------------------------------

            noise_map = get_noise_map(orig)
            noisemap_filepath = os.path.join('noisemap_results/IMD2020', os.path.basename(orig) + '_noisemap.png')
            cv2.imwrite(noisemap_filepath, noise_map)#save map

            #second map
            originalBGR = cv2.imread(orig)
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

            noisemap_2_filepath = os.path.join('noisemap_results/IMD2020', os.path.basename(orig) + '_noisemap_2.png')
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
                "image_name": orig,
                "image noise name": noisemap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "original"
            }
            
            with open('results_noise_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')


        #manipulated images -------------------------------------------------------------------------------------

        
        n_man_im = 0
        for man_im in manipulated_images:
            #get corresponding mask path for the manipulated image
            mask_man_im = masks[n_man_im]

            #ghostmap analysis for manipulated image
            outputmaps = process_ghostmaps(man_im, Q_min, Q_max, Q_step, shift_x, shift_y)
            #save total outputmap as image, including original:
            save_ghostmaps(man_im, outputmaps)
            #drop edge cases from ghost map output, so statistical difference is calculated only for non-edge cases:
            outputmaps_filtered, edge = drop_edge_maps(outputmaps)

            results_stat_difference = []
            n_outputmap_filtered = 0

            #resize mask to outputmap size
            mask = cv2.imread(mask_man_im, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, (outputmaps_filtered.shape[1], outputmaps_filtered.shape[0]), interpolation=cv2.INTER_AREA)
            for i in range(n_ghostmaps-2*edge):
                outmap = outputmaps_filtered[:,:, i]
                #list values inside mask and values outside mask
                values_inside_mask = outmap[resized_mask > 0.5]
                values_outside_mask = outmap[resized_mask <= 0.5]
                
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
                "image name": man_im,
                "image outputmaps name": os.path.join('ghostmaps_results/IMD2020', os.path.basename(man_im) + '_ghostmaps.png'),
                "largest med diff map": (Q_min + largest_med_diff_mapnr * Q_step),
                "median inside mask": med_inside_mask,
                "median outside mask": med_outside_mask,
                "absolute median difference": largest_med_diff,
                "map largest k stat": (Q_min + largest_k_diff_mapnr * Q_step),
                "largest k stat diff": largest_k_diff,
                "type": "manipulated"
            }
            
            
            with open('results_ghostmaps_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')
            
            #probability -----------------------------------------------------------------------------------------------
            #construct mask probability map 3x3
            mask = cv2.imread(mask_man_im, cv2.IMREAD_GRAYSCALE)
            resized_mask = mask[1:mask.shape[0]-1,1:mask.shape[1]-1]


            #probabilitymap 3x3 + save --------------------------------------------------------------------------
            probability_map_3x3 = calculate_probability_map_3x3(man_im)
            probmap_filepath = os.path.join('probability_map_3x3_results/IMD2020', os.path.basename(man_im) + '_probmap.png')
            probability_map_3x3 = probability_map_3x3 *255 #probmap_filepath = normalized between 0 and 1, so rescale to 0 - 255
            cv2.imwrite(probmap_filepath, probability_map_3x3)#save map

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
                "image_name": man_im,
                "image prob name": probmap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "manipulated"
            }
            

            with open('results_probability_3x3_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

            #noise --------------------------------------------------------------------------------------------------

            noise_map = get_noise_map(man_im)
            noisemap_filepath = os.path.join('noisemap_results/IMD2020', os.path.basename(man_im) + 'noisemap.png')
            cv2.imwrite(noisemap_filepath, noise_map)#save map

            #second map
            originalBGR = cv2.imread(man_im)
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

            noisemap_2_filepath = os.path.join('noisemap_results/IMD2020', os.path.basename(man_im) + '_noisemap_2.png')
            plt.savefig(noisemap_2_filepath)

            #construct mask noise manipulated
            mask = cv2.imread(mask_man_im, cv2.IMREAD_GRAYSCALE)
            resized_mask = cv2.resize(mask, (noise_map.shape[1], noise_map.shape[0]), interpolation=cv2.INTER_AREA)

            values_inside_mask = noise_map[resized_mask > 0.5]
            values_outside_mask = noise_map[resized_mask <= 0.5]

            #calculate statistical difference using Kolmogorov-Smirnov statistic
            k_stat, _ = ks_2samp(values_inside_mask, values_outside_mask, nan_policy='omit')
                
            #calculate medians inside and outside mask
            median_inside_mask = np.nanmedian(values_inside_mask)
            median_outside_mask = np.nanmedian(values_outside_mask)

            #save data in file
            result_data = {
                "image_name": man_im,
                "image noise name": noisemap_filepath,
                "median inside mask": median_inside_mask,
                "median outside mask": median_outside_mask,
                "absolute median difference": abs(median_inside_mask - median_outside_mask),
                "k stat diff": k_stat,
                "type": "manipulated"
            }
            
            
            with open('results_noise_IMD2020.json', 'a') as f:
                json.dump(result_data, f)
                f.write('\n')

            n_man_im += 1

   


        #AI predictions --------------------------------------------------------------
        #mm-fusion
        #mm-fusion expects images as absolute paths
        try:
            list_image_paths = orig_images + manipulated_images
            base_path = os.getcwd()
            list_absolute_image_paths = [os.path.join(base_path, relative_path) for relative_path in list_image_paths]
            detection_scores, map_locations = process_images(list_absolute_image_paths, 'MM-fusion_results/IMD2020/')
            #original ------------------------------------------------------------------------------------
            mm_counter = 0
            for orig in orig_images:
                
                #map mmfusion
                fusion_map = cv2.imread(map_locations[mm_counter], cv2.IMREAD_GRAYSCALE)

                #mask
                mask = np.zeros(fusion_map.shape)
                center_x, center_y = original_im.shape[0] // 2, original_im.shape[1] // 2
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
                    "image_name": orig,
                    "image output map location": map_locations[mm_counter],
                    "median inside mask": median_inside_mask,
                    "median outside mask": median_outside_mask,
                    "absolute median difference": abs(median_inside_mask - median_outside_mask),
                    "k stat diff": k_stat,
                    "detection score AI-model": detection_scores[mm_counter],
                    "type": "original"
                }
                

                with open('results_MM-fusion_IMD2020.json', 'a') as f:
                    json.dump(result_data, f)
                    f.write('\n')
                mm_counter += 1

            n_man_im = 0
            for manip in manipulated_images:
                
                mask = cv2.imread(masks[n_man_im], cv2.IMREAD_GRAYSCALE)

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
                    "image_name": manip,
                    "image output map location": map_locations[mm_counter],
                    "median inside mask": median_inside_mask,
                    "median outside mask": median_outside_mask,
                    "absolute median difference": abs(median_inside_mask - median_outside_mask),
                    "k stat diff": k_stat,
                    "detection score AI-model": detection_scores[mm_counter],
                    "type": "manipulated"
                }
                

                with open('results_MM-fusion_IMD2020.json', 'a') as f:
                    json.dump(result_data, f)
                    f.write('\n')
                mm_counter += 1
                n_man_im +=1
            
        except:
            with open('error_log.json', 'a') as f:
                json.dump("Error occured in map for MM-fusion:" + submap , f)
                f.write('\n')


        #DFAD ------------------------------------------------
        try:
            list_image_paths = orig_images + manipulated_images
            detection_scores = process_images_DFAD(list_image_paths)
            #all ------------------------------------------------------------------------------------
            mm_counter = 0
            for orig in orig_images:

                #save data in file
                result_data = {
                    "image_name": orig,
                    "prediction score AI-model": detection_scores[mm_counter].item(),
                    "type": "original"
                }

                with open('results_DFAD2023_IMD2020.json', 'a') as f:
                    json.dump(result_data, f)
                    f.write('\n')
                mm_counter += 1

            n_man_im = 0
            for manip in manipulated_images:

                #save data in file
                result_data = {
                    "image_name": manip,
                    "prediction score AI-model": detection_scores[mm_counter].item(),
                    "type": "manipulated"
                }
                

                with open('results_DFAD2023_IMD2020.json', 'a') as f:
                    json.dump(result_data, f)
                    f.write('\n')
                mm_counter += 1
                n_man_im +=1
            
        except:
            with open('error_log.json', 'a') as f:
                json.dump("Error occured in map, for DFAD2023:" + submap , f)
                f.write('\n')
    

    

    
